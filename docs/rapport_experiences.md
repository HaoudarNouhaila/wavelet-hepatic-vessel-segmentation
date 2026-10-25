# Rapport d'expériences

Ce document retrace les expériences réellement effectuées, dans l'ordre, avec leurs conditions et
leurs résultats. Les valeurs numériques proviennent des fichiers de `results/metrics/` et des
dossiers `experiments/`.

## 1. Environnement

- Windows 11, Python 3.10.10, PyTorch 2.8.0 (CPU), MONAI 1.6.1, PyWavelets 1.8.0 (versions complètes
  dans `results/environment.txt` et `requirements.txt`).
- Processeur Intel Core i5-1135G7 (4 cœurs, 8 threads), 16 Go de RAM, carte graphique intégrée
  Intel Iris Xe : pas de CUDA, tous les calculs sont faits sur CPU.
- Débit réseau mesuré : 0,1 à 0,45 Mo/s, d'où le téléchargement partiel décrit dans
  `docs/donnees.md`.
- WSL Ubuntu est installé sur la machine mais n'a pas été nécessaire : le code ne dépend que de
  Python et fonctionne de la même façon sous Windows et Linux.

Problèmes rencontrés lors de l'installation :

- l'environnement Python global contenait des versions incompatibles (pandas 1.5 et SciPy 1.11
  compilés pour NumPy 1, avec NumPy 2.2) : j'ai créé un environnement virtuel isolé ;
- SimpleITK ne se chargeait pas depuis un environnement placé dans un dossier au chemin très long
  (« The filename or extension is too long », limite de Windows pour le chargement des DLL) :
  l'environnement a été déplacé dans un dossier au chemin court.

## 2. Mesure des temps de calcul (essai préliminaire)

Avant les expériences, j'ai entraîné le U-Net 2D pendant 3 époques sur les 10 premiers volumes
téléchargés (8 en entraînement, 2 en validation), uniquement pour vérifier le pipeline sur des
données réelles et mesurer les temps. Cet essai n'est pas utilisé dans les résultats. Il a donné :

- environ 0,16 s par patch de 192 x 192 en entraînement (lot de 8, 4 threads) ;
- environ 0,3 s par coupe 512 x 512 en inférence ;
- deux entraînements en parallèle avec 2 threads chacun vont légèrement plus vite au total qu'un
  seul entraînement avec 4 threads (0,15 s contre 0,16 s par patch).

Ces mesures ont fixé le budget : 960 patches par époque, 25 époques au plus, validation sur une
coupe sur trois, deux entraînements simultanés.

## 3. Modification du protocole avant les expériences finales

Lors du premier lancement des entraînements finaux, le Wavelet U-Net (SWT) ne prédisait encore
aucun vaisseau après deux époques (Dice de validation de 0), alors que le U-Net standard atteignait
0,28. Avec une patience de 7 époques et sans minimum, l'arrêt anticipé aurait pu stopper ce modèle
avant qu'il apprenne. J'ai vérifié que la normalisation des sous-bandes n'était pas en cause
(écart-type proche de 1 par canal, valeurs extrêmes limitées aux bords du corps et des os), puis
j'ai ajouté un minimum de 12 époques avant tout arrêt anticipé, pour tous les modèles, et relancé
les entraînements depuis le début. Les premières époques relancées sont identiques au premier
lancement, ce qui confirme que les tirages sont bien reproductibles.

## 4. Entraînements

Deux entraînements simultanés (2 threads chacun) ; durée médiane d'une époque d'environ 5,5 min.
Les durées brutes de `summary.json` incluent plusieurs mises en veille de la machine (plus de 13 h
au total pour la variante SWT seed 1) et ne sont pas reportées ici.

| Expérience | Meilleure époque | Époques effectuées | Dice vaisseaux en validation |
|---|---|---|---|
| U-Net 2D, seed 0 | 12 | 19 | 0,531 |
| U-Net 2D, seed 1 | 18 | 25 | 0,475 |
| Wavelet U-Net SWT, seed 0 | 12 | 19 | 0,516 |
| Wavelet U-Net SWT, seed 1 | 22 | 25 | 0,500 |
| Wavelet U-Net multi-échelle, seed 0 | 10 | 17 | 0,512 |
| Wavelet U-Net multi-échelle, seed 1 | 17 | 24 | 0,501 |
| U-Net 3D, seed 0 | 32 | 40 | 0,442 |

Une troisième seed a été lancée pour le U-Net 2D puis arrêtée à l'époque 3 : avec les mises en
veille successives, les trois entraînements supplémentaires et leurs évaluations auraient ajouté
plus de dix heures de calcul. Ce run incomplet a été supprimé et n'est utilisé nulle part.

## 5. Évaluation sur l'ensemble de test

20 volumes, métriques calculées par volume sur la grille d'origine (`results/metrics/test_per_volume.csv`,
résumés dans `test_per_seed.csv`, `test_summary.csv` et `test_wilcoxon_vs_unet2d.csv`). Inférence :
environ 35 s par volume pour un modèle 2D et 18 s pour le U-Net 3D (2 threads).

Résultats principaux (vaisseaux, images originales) : voir le tableau du README. Points saillants :

- avec la seed 0, le U-Net 2D a le meilleur Dice (0,545 contre 0,527 et 0,529, p = 0,07 et 0,60) ;
  avec la seed 1, les variantes ondelettes le dépassent (0,551 et 0,564 contre 0,493, sur 18 et
  17 volumes sur 20). Le sens de la différence dépend de l'entraînement ;
- la variante multi-échelle a un HD95 plus faible que le U-Net pour les deux seeds (- 8,2 mm,
  p = 0,03 ; - 20,6 mm, p = 0,0003) et moins de composantes isolées (59 contre 86, 104 contre 121) ;
- rappel de la ligne centrale : 0,43 à 0,57 (rayon < 1,5 mm), 0,72 à 0,82 (1,5 à 3 mm), 0,65 à
  0,74 (> 3 mm), sans ordre constant entre les modèles ;
- U-Net 3D : Dice 0,441, clDice 0,490, HD95 78 mm, 186 composantes isolées par volume ;
- tumeurs : Dice de 0,17 à 0,33 selon le modèle et la seed.

## 6. Étude de la résolution

Conditions évaluées : seed 0 de chaque modèle 2D dans les 5 conditions ; seed 1 et U-Net 3D sur
les images originales et les sous-échantillonnages x2 et x4. Le facteur 3, prévu au départ, a été
retiré avant l'évaluation pour tenir dans le budget de calcul.

Dice relatif à l'image originale (médiane sur les volumes, seed 0,
`results/metrics/test_dice_relatif_resolution.csv`) :

| | flou 1 mm | x2 | x2 + bruit 20 HU | x4 |
|---|---|---|---|---|
| U-Net 2D | 1,016 | 1,023 | 0,843 | 0,994 |
| Wavelet U-Net SWT | 0,946 | 0,978 | 0,861 | 0,889 |
| Wavelet U-Net multi-échelle | 0,968 | 0,996 | 0,916 | 0,916 |
| U-Net 3D | - | 0,975 | - | 0,794 |

Avec la seed 1, l'ordre est inversé au facteur 4 (moyennes sur les volumes : - 12 % pour le U-Net,
- 3,5 % pour la variante SWT, - 5 % pour la variante multi-échelle). Dans toutes les configurations,
le facteur 4 fait surtout chuter le rappel des vaisseaux de rayon inférieur à 1,5 mm (0,27 à 0,41),
alors que celui des vaisseaux de plus de 3 mm change peu.

## 7. Recalage sur transformations synthétiques (extension facultative)

Expérience indépendante de la segmentation (`scripts/registration_demo.py`, résultats dans
`results/metrics/registration_synthetic.csv`). Volume : hepaticvessel_124 (premier cas de
validation). Une transformation connue est appliquée au volume, puis estimée par SimpleITK
(information mutuelle de Mattes, 5 % des voxels échantillonnés, descente de gradient à pas
régulier, trois niveaux de résolution). L'erreur (TRE) est mesurée sur 500 points tirés dans les
vaisseaux.

| transformation de référence | TRE avant | TRE moyenne après | TRE max après | temps |
|---|---|---|---|---|
| translation (6, -4, 5) mm | 8,77 mm | 0,26 mm | 0,26 mm | 47 s |
| rigide (rotations 2°, -3°, 6° et translation) | 11,05 mm | 0,03 mm | 0,04 mm | 45 s |
| affine (rotation, translation, échelles 0,97 à 1,04) | 10,23 mm | 0,31 mm | 0,52 mm | 86 s |

Dans les trois cas, l'erreur résiduelle est inférieure à la taille d'un voxel. La translation pure
est un peu moins précise que la transformation rigide : l'optimiseur s'arrête quand le pas devient
trop petit, sur un plateau de la métrique probablement lié à l'interpolation linéaire ; je n'ai pas
cherché à ajuster ses paramètres. Ce cas est volontairement favorable : même volume, même
modalité, transformation exactement représentable par le modèle de transformation, pas de bruit ni
de déformation. Il ne représente pas un recalage entre deux examens, ni un recalage CT / micro-CT.
