# Méthodologie

## 1. Point de départ

J'avais auparavant reproduit un projet de segmentation cardiaque CT par U-Net (TensorFlow / Keras,
coupes PNG 512 x 512, perte Dice). En relisant ce code pour préparer ce projet, j'ai gardé la
structure générale en quatre étapes (préparation, entraînement, évaluation, test) et l'architecture
U-Net à quatre sous-échantillonnages. J'ai en revanche changé plusieurs points qui posent problème
pour une évaluation rigoureuse :

- la séparation entraînement / validation y est faite au niveau des coupes : des coupes voisines
  d'un même patient peuvent se retrouver des deux côtés. Ici, la séparation se fait au niveau du
  patient et un ensemble de test distinct n'est utilisé qu'à la fin ;
- les images y sont lues en PNG et normalisées par 255, ce qui perd les unités Hounsfield. Ici, les
  volumes NIfTI sont lus directement et normalisés à partir des HU ;
- les métriques y sont moyennées par coupe, avec un score de 1 attribué aux coupes sans structure
  (`zero_division=1`), ce qui gonfle les résultats. Ici, les métriques sont calculées par volume, et un
  cas non défini vaut NaN au lieu de 1 ;
- la fonction d'augmentation contenait un `break` qui arrêtait la boucle après la première image ;
- le masque y était redimensionné par interpolation bilinéaire puis seuillé. Ici, aucune
  interpolation n'est appliquée aux masques (et le plus proche voisin serait utilisé si besoin).

## 2. Tâche

Segmentation 2D (coupe par coupe) en trois classes : fond, vaisseaux hépatiques, tumeurs. Les
vaisseaux sont la classe étudiée ; les tumeurs sont gardées dans la sortie pour respecter
l'annotation, et leurs métriques sont rapportées séparément.

## 3. Modèles

### U-Net 2D de référence

Implémentation PyTorch dans `src/models/unet2d.py` : encodeur à cinq étages (16, 32, 64, 128, 256
filtres), chaque étage composé de deux convolutions 3 x 3 + BatchNorm + ReLU, max-pooling 2 x 2,
décodeur symétrique avec convolutions transposées 2 x 2 et connexions de saut par concaténation,
couche de sortie 1 x 1 à trois canaux (logits). 1 942 323 paramètres. J'ai choisi 16 filtres au
premier étage au lieu des 64 de l'article original pour que l'entraînement reste possible sur CPU.

### Wavelet-Guided U-Net, fusion en entrée (`wavelet_unet.yaml`)

L'entrée contient la coupe CT normalisée et ses quatre sous-bandes de Haar (LL, LH, HL, HH) obtenues
par la transformée en ondelettes stationnaire (SWT, non décimée) de PyWavelets. La SWT donne des
sous-bandes de même taille que l'image : chaque coefficient est aligné sur un pixel, sans
sur-échantillonnage ni recadrage. Chaque sous-bande est standardisée avec une moyenne et un
écart-type estimés sur des coupes d'entraînement. Seul le nombre de canaux d'entrée change
(5 au lieu de 1) : 1 942 899 paramètres.

### Wavelet-Guided U-Net, injection multi-échelle (`wavelet_unet_multiscale.yaml`)

L'entrée reste la coupe CT seule. Les sous-bandes de la DWT de Haar au niveau 1 (taille H/2) sont
concaténées aux caractéristiques de l'étage 1 de l'encodeur, après le max-pooling, et celles du
niveau 2 (DWT de la sous-bande LL1, taille H/4) à l'étage 2. La décimation de la DWT correspond
exactement au sous-échantillonnage du U-Net, ce qui aligne les résolutions sans interpolation.
1 945 779 paramètres.

Dans les deux variantes, la transformée est calculée en prétraitement avec PyWavelets (NumPy), après
l'augmentation de données, sur l'image augmentée. Elle n'est pas différentiable et n'est pas
apprise : il ne s'agit pas d'une couche DWT intégrée au réseau.

Points de vigilance traités :

- alignement des résolutions : SWT pleine résolution pour l'entrée, niveaux DWT de taille H/2^k pour
  l'injection ;
- bords : les caractéristiques sont calculées sur un recadrage plus grand de 8 pixels de chaque côté,
  puis cette marge est retirée, ce qui supprime l'effet de l'extension périodique aux bords ;
  à l'inférence, elles sont calculées sur la coupe entière ;
- grille de la DWT : les recadrages commencent sur des coordonnées multiples de 4, si bien qu'en
  l'absence d'augmentation la DWT de Haar d'un patch est exactement le recadrage de la DWT de la
  coupe entière (vérifié par un test) ;
- labels : la transformée n'est jamais appliquée aux masques et n'utilise aucune information issue
  des labels.

### U-Net 3D (extension)

`monai.networks.nets.UNet` en 3D, canaux (16, 32, 64, 128), un premier sous-échantillonnage limité au
plan (stride (1, 2, 2)) à cause de l'anisotropie des volumes, puis (2, 2, 2). Entraînement par
patches de 16 x 128 x 128 voxels, inférence par fenêtre glissante (32 x 256 x 256, recouvrement 25 %,
pondération gaussienne). 600 680 paramètres.

## 4. Entraînement

Les trois modèles 2D partagent `configs/base_2d.yaml` ; seule la section `model` diffère.

- Patches de 192 x 192 pixels tirés dans les coupes des 40 volumes d'entraînement : avec une
  probabilité de 0,67 le patch est centré sur un voxel annoté (vaisseau ou tumeur), sinon sur un
  point quelconque du volume, y compris hors du foie.
- 960 patches par époque, lots de 8.
- Augmentations : rotation (± 15°) et mise à l'échelle (0,85 à 1,15) avec probabilité 0,3,
  retournements horizontal et vertical, variation linéaire d'intensité, bruit gaussien. Interpolation
  linéaire pour l'image et plus proche voisin pour le masque.
- Perte : Dice loss (classes vaisseau et tumeur, calculée sur le lot entier) + cross-entropy, avec des
  poids égaux. La Dice loss compense le fort déséquilibre des classes (les vaisseaux occupent une
  petite fraction des voxels, voir `docs/donnees.md`).
- Optimiseur AdamW (taux d'apprentissage 1e-3, weight decay 1e-4), décroissance cosinus jusqu'à
  1e-5 sur 25 époques au plus.
- Validation à chaque époque sur les 8 volumes de validation (une coupe sur trois, sous-échantillonnage
  régulier qui ne dépend pas des labels, pour limiter le coût sur CPU), avec la perte et le Dice par
  volume. Le meilleur modèle est celui qui maximise le Dice moyen des vaisseaux en validation ;
  l'entraînement s'arrête après 7 époques sans amélioration, mais jamais avant 12 époques.
  J'ai ajouté ce minimum après un premier lancement : le Wavelet U-Net (SWT) ne prédisait encore
  aucun vaisseau après deux époques alors que le U-Net en était à un Dice de 0,28 en validation. Sans
  minimum, l'arrêt anticipé aurait pu interrompre ce modèle avant qu'il apprenne, ce qui aurait faussé
  la comparaison. Les deux entraînements ont été relancés depuis le début avec la même règle pour
  tous les modèles ; ce choix ne dépend d'aucun résultat de test.
- Ces valeurs ont été fixées avant les expériences, à partir d'un essai de 3 époques sur 10 volumes
  servant uniquement à mesurer les temps de calcul (environ 0,16 s par patch en entraînement et
  0,3 s par coupe 512 x 512 en inférence sur le CPU utilisé). Elles sont identiques pour tous les
  modèles et n'ont pas été ajustées en fonction des résultats.
- Reproductibilité : chaque patch est tiré à partir de (seed, époque, indice). Pour une même seed,
  les trois modèles voient exactement les mêmes patches et les mêmes augmentations (vérifié par un
  test). Les configurations effectives sont copiées dans chaque dossier d'expérience.

## 5. Évaluation

L'ensemble de test (20 volumes) n'est utilisé qu'une fois les modèles choisis sur la validation.
Chaque volume est rechargé depuis le NIfTI d'origine, prétraité, segmenté coupe par coupe sur la
coupe entière (512 x 512), et comparé au masque d'origine.

Métriques calculées par volume et par classe (`src/evaluation/metrics.py`) :

- Dice, IoU, précision, rappel et F1 au niveau des voxels. Pour une segmentation binaire, F1 et Dice
  sont égaux par définition ; je recalcule le F1 à partir de la précision et du rappel comme contrôle ;
- HD95 en millimètres : maximum des deux distances de surface dirigées au 95e percentile, avec
  l'espacement réel des voxels. Les distances entre voxels de surface sont obtenues exactement par un
  arbre k-d, ce qui équivaut à une transformée de distance avec beaucoup moins de mémoire (même
  définition que MONAI, résultat identique vérifié par un test) ;
- clDice (Shit et al., 2021) : moyenne harmonique de la part du squelette prédit contenue dans la
  référence et de la part du squelette de référence couverte par la prédiction. Les squelettes sont
  calculés en 3D (méthode de Lee, scikit-image) sur la grille des voxels. Des tests sur des tubes
  synthétiques vérifient qu'un vaisseau prédit trop épais garde un clDice proche de 1, et qu'une
  petite branche manquée fait baisser le clDice nettement plus que le Dice.

Analyse des vaisseaux (`src/evaluation/vessel_analysis.py`) :

- rappel de la ligne centrale par classe de rayon : le rayon local est la distance (mm) entre un point
  du squelette de référence et le bord du vaisseau ; classes 0-1,5 mm, 1,5-3 mm et plus de 3 mm ;
- composantes connexes (connexité 26) : nombre, part des composantes de référence touchées,
  composantes prédites sans aucun recouvrement (faux positifs isolés), part du volume prédit dans la
  plus grande composante ;
- cartes d'erreurs (vrais positifs, faux positifs, faux négatifs).

Les résultats sont d'abord enregistrés par volume (`results/metrics/test_per_volume.csv`), puis
résumés pour chaque seed. Les comparaisons entre modèles utilisent un test de Wilcoxon apparié sur
les volumes de test, fait séparément pour chaque seed (modèles de même seed) puis sur les valeurs
moyennées sur les seeds. Ce test ne mesure que la variabilité entre volumes : il ne dit rien de la
variabilité entre entraînements, qui s'est révélée du même ordre que les écarts entre modèles. Avec
20 volumes, une absence de différence significative ne prouve pas non plus que les modèles sont
équivalents.

Deux seeds (0 et 1) ont été entraînées pour chaque modèle 2D, et une seule pour le U-Net 3D.

## 6. Étude de la résolution

À partir des volumes de test originaux, j'ai créé des images dégradées dans le plan de coupe
(`src/preprocessing/degradation.py`), appliquées aux HU avant la normalisation :

- flou gaussien de sigma 1 mm ;
- sous-échantillonnage d'un facteur 2 ou 4 (filtre anti-repliement gaussien, décimation, puis
  retour sur la grille d'origine par interpolation spline cubique) ;
- sous-échantillonnage d'un facteur 2 suivi d'un bruit gaussien de 20 HU.

Toutes les conditions ont été évaluées pour la seed 0 des modèles 2D ; la seed 1 et le U-Net 3D
ont été évalués sur les images originales et les facteurs 2 et 4. Un facteur 3 était prévu ; je
l'ai retiré avant l'évaluation pour tenir dans le budget de calcul.

Le masque de référence reste celui d'origine : on mesure la capacité du modèle à retrouver
l'anatomie à partir d'une image moins résolue. Les modèles ne sont pas réentraînés sur ces images.
Ces images simulées servent uniquement à étudier la sensibilité à la résolution : elles ne
correspondent ni à de vraies acquisitions à une autre résolution, ni à des paires CT / micro-CT.

## 7. Recalage (extension facultative)

`scripts/registration_demo.py` applique à un volume CT des transformations connues (translation,
rigide, affine), puis les estime par recalage SimpleITK (information mutuelle de Mattes, descente de
gradient à pas régulier, trois niveaux de résolution). L'erreur est mesurée par la TRE sur 500 points
tirés dans les vaisseaux. C'est un exercice de validation sur un même volume, indépendant du
pipeline de segmentation.
