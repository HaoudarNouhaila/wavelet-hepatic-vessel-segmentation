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

## 8. Reprise sur GPU (Kaggle)

### Conditions

- Notebook `notebooks/05_kaggle_gpu.ipynb` exécuté en privé sur Kaggle (Tesla T4, 4 processeurs,
  31 Go de RAM, Python 3.13, PyTorch 2.11) ; les 65 tests passent sur cet environnement. Le journal
  nettoyé de l'exécution est dans `results/kaggle/journal_execution_kaggle.txt`.
- Même séparation au niveau patient (seed 42), avec tous les cas : 212 / 45 / 46. Les 20 volumes de
  test CPU font partie des 46.
- Mêmes architectures, perte, optimiseur et augmentations que sur CPU. Réglages propres au GPU
  (`configs/kaggle/gpu_2d.yaml`), identiques pour les trois modèles 2D : lots de 16, 4 000 patches
  par époque, 40 époques au plus, patience de 10 époques et minimum de 15, validation sur une coupe
  sur deux des 45 volumes, 4 processus de chargement. U-Net 3D (`configs/kaggle/gpu_3d.yaml`) :
  patches de 24 x 160 x 160, lots de 4, 800 patches par époque, 60 époques au plus.
- Évaluation : images originales et facteurs 2 et 4 pour tous les modèles ; flou de 1 mm et facteur
  2 avec bruit pour la seed 0, comme sur CPU.

Un premier lancement a échoué après 34 minutes : le dossier `src/data/` n'était pas sur GitHub, car
la règle `data/` du `.gitignore` l'excluait en plus du dossier de données. La règle a été limitée à
`/data/`, un test vérifie maintenant qu'aucun fichier source n'est ignoré, et le dépôt a été
vérifié à partir d'un clone vierge avant le second lancement.

Durées du second lancement : téléchargement de l'archive 15 min, prétraitement des 303 cas 5 min,
neuf entraînements 2D de 16 à 28 min chacun (3,1 h au total), U-Net 3D 86 min, évaluation 62 min
(une condition par processus), soit 5,9 h de session.

### Entraînements

| Expérience | Meilleure époque / époques | Dice vaisseaux en validation |
|---|---|---|
| U-Net 2D, seeds 0 / 1 / 2 | 8 / 18, 10 / 20, 16 / 26 | 0,533, 0,542, 0,441 |
| Wavelet U-Net SWT, seeds 0 / 1 / 2 | 5 / 15, 10 / 20, 7 / 17 | 0,539, 0,528, 0,584 |
| Wavelet U-Net multi-échelle, seeds 0 / 1 / 2 | 6 / 16, 10 / 20, 7 / 17 | 0,514, 0,530, 0,516 |
| U-Net 3D, seed 0 | 56 / 60 | 0,567 |

Les modèles 2D atteignent leur meilleur score de validation très tôt (5 à 16 époques) puis
plafonnent. Le U-Net 3D progresse jusqu'à la fin du budget.

### Résultats sur les 46 volumes de test (vaisseaux, images originales)

| Modèle | Seed | Dice | clDice | HD95 (mm) | Précision | Rappel |
|---|---|---|---|---|---|---|
| U-Net 2D | 0 / 1 / 2 | 0,538 / 0,551 / 0,490 | 0,580 / 0,586 / 0,528 | 50,3 / 43,6 / 77,3 | 0,606 / 0,666 / 0,391 | 0,557 / 0,512 / 0,751 |
| Wavelet SWT | 0 / 1 / 2 | 0,535 / 0,538 / 0,592 | 0,576 / 0,585 / 0,633 | 46,3 / 66,6 / 35,8 | 0,624 / 0,591 / 0,641 | 0,535 / 0,557 / 0,611 |
| Wavelet multi-échelle | 0 / 1 / 2 | 0,540 / 0,546 / 0,555 | 0,597 / 0,580 / 0,588 | 40,4 / 58,3 / 69,6 | 0,589 / 0,536 / 0,526 | 0,578 / 0,623 / 0,646 |
| U-Net 3D | 0 | 0,582 | 0,639 | 36,9 | 0,663 | 0,574 |

- Le U-Net 2D seed 2 est un entraînement nettement moins bon que les deux autres : il sur-segmente
  (précision 0,39, rappel 0,75). C'est lui qui donne l'avantage moyen aux variantes ondelettes ;
  avec les seeds 0 et 1, elles ne font pas mieux que le U-Net (seed 0 : p = 0,75 et 0,86 pour le
  Dice ; seed 1 : SWT - 0,013, p = 0,01 ; multi-échelle - 0,004, p = 0,11).
- La réduction du HD95 observée sur CPU pour la variante multi-échelle ne se retrouve pas : elle
  est vraie pour les seeds 0 et 2 (- 10,0 et - 7,7 mm) mais pas pour la seed 1 (+ 14,7 mm, p = 0,0001).
- Sur les 20 volumes de test communs, le passage de 40 à 212 volumes d'entraînement fait gagner
  + 0,019 de Dice au U-Net, + 0,024 à la variante SWT et + 0,004 à la variante multi-échelle
  (moyennes des seeds), contre + 0,150 au U-Net 3D.
- Robustesse au facteur 4 (Dice relatif à l'image originale) : U-Net 2D 0,91 / 0,98 / 0,99, SWT
  0,82 / 0,95 / 0,94, multi-échelle 0,84 / 0,96 / 0,96, U-Net 3D 0,91. Pour chacune des trois seeds,
  les deux variantes ondelettes perdent davantage que le U-Net de même seed. Avec le flou de 1 mm
  (seed 0) : 0,96 pour le U-Net, 0,92 et 0,91 pour les variantes ondelettes, 0,96 pour le U-Net 3D.
- Petits vaisseaux : rappel de la ligne centrale de 0,32 à 0,59 (rayon < 1,5 mm) selon le modèle
  et la seed, 0,22 à 0,41 au facteur 4, alors que les vaisseaux de plus de 3 mm restent au-dessus
  de 0,60.
- Tumeurs : Dice de 0,30 à 0,42 selon le modèle et la seed (0,17 à 0,33 sur CPU).

### Conclusion de la reprise sur GPU

Avec cinq fois plus de données d'entraînement et une seed de plus, la conclusion sur la question de
recherche ne change pas : les sous-bandes de Haar, en entrée ou injectées dans l'encodeur,
n'améliorent pas de façon robuste la segmentation des vaisseaux par rapport au U-Net standard. Le
seul effet constant sur les trois seeds est défavorable aux ondelettes : une moins bonne
robustesse à la baisse de résolution, cohérente avec la disparition des coefficients de détail sur
les images dégradées. Le gain le plus net vient du passage au U-Net 3D entraîné sur toutes les
données.
