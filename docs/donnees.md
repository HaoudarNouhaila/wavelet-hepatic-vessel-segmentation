# Données : Medical Segmentation Decathlon, Task08_HepaticVessel

## Provenance et licence

Le jeu de données provient du Medical Segmentation Decathlon (MSD), tâche 8 « Hepatic Vessel ».
D'après le fichier `dataset.json` de l'archive (version 1.1 du 14/08/2018), les données viennent du
Memorial Sloan Kettering Cancer Center et sont distribuées sous licence CC-BY-SA 4.0. La page
officielle demande de citer Simpson et al. (2019) ; l'article de référence du challenge est Antonelli
et al. (2022).

J'ai utilisé le miroir AWS indiqué sur la page officielle :
`https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar` (9 347 254 784 octets).

## Contenu vérifié de l'archive

J'ai lu la totalité des en-têtes tar de l'archive (1 502 entrées) sans la télécharger, à l'aide de
requêtes HTTP partielles (`src/data/remote_tar.py`). L'archive contient :

- `imagesTr/` : 303 volumes CT annotés (environ 19,5 Mo chacun en .nii.gz, 5,9 Go au total) ;
- `labelsTr/` : 303 masques ;
- `imagesTs/` : 140 volumes de test sans annotation (3,0 Go), non utilisés ici ;
- `dataset.json` ;
- un fichier caché macOS `._*` (AppleDouble) pour chaque fichier et dossier. Ces fichiers ne sont
  pas des NIfTI et sont ignorés au chargement.

Le champ `numTest` de `dataset.json` vaut 140 et l'archive contient bien 140 fichiers dans
`imagesTs/`, mais la liste `test` du même fichier n'en recense que 139. Cela n'a pas d'effet sur le
projet puisque seuls les cas annotés sont utilisés.

Labels (d'après `dataset.json`) : 0 = fond, 1 = vaisseaux (« Vessel »), 2 = tumeur (« Tumour »).

## Géométrie des volumes

Les en-têtes NIfTI des 443 volumes ont été lus en décompressant uniquement les premiers kilo-octets
de chaque fichier (`scripts/download_data.py`, résultat dans `results/metrics/dataset_headers.csv`).

| | coupes par volume | espacement dans le plan (mm) | épaisseur de coupe (mm) |
|---|---|---|---|
| 303 volumes annotés | 24 à 181 (médiane 49, moyenne 69,7) | 0,57 à 0,98 (moyenne 0,80) | 0,8 à 8,0 (médiane 5,0) |

Toutes les coupes font 512 x 512 pixels. L'espacement varie d'un patient à l'autre : les volumes
n'ont donc pas la même résolution physique, et ils sont très anisotropes (la plupart ont des coupes
de 5 mm pour 0,8 mm dans le plan).

## Intensités et labels (68 cas téléchargés)

Résultats de `scripts/inspect_dataset.py` (`results/metrics/dataset_inspection.csv`, figures dans
`results/figures/dataset/`) :

- les 68 volumes sont stockés en orientation RAS ; les intensités vont de -1024 HU à 1147-3072 HU ;
- les vaisseaux ont une intensité moyenne de 157 HU (écart-type entre cas : 29 HU), les tumeurs de
  91 HU : les images sont injectées (phase portale) et les vaisseaux sont plus brillants que le
  parenchyme hépatique, mais les distributions se recouvrent en partie ;
- les vaisseaux n'occupent que 0,10 % des voxels en moyenne (0,02 % à 0,25 %), soit 36,5 mL par
  volume en moyenne (11 à 87 mL). Ce déséquilibre justifie la Dice loss et l'échantillonnage des
  patches autour des structures annotées ;
- les tumeurs sont très variables : de 0,5 mL à 2 122 mL (médiane 45 mL) ;
- les vaisseaux apparaissent sur 39 coupes par volume en moyenne et forment en moyenne
  49 composantes connexes (7 à 189) en connexité 26. Cette fragmentation vient en partie de
  l'épaisseur des coupes (5 mm le plus souvent) : un petit vaisseau oblique ne se recouvre pas
  forcément d'une coupe à l'autre ;
- sur certaines coupes, des branches fines visibles à l'image ne semblent pas annotées. Je ne l'ai
  pas quantifié, mais cela peut pénaliser un modèle qui les segmente (faux positifs apparents).

Tous les contrôles de chargement (dimensions et affines identiques entre image et masque, labels
dans {0, 1, 2}) sont passés sans erreur pour les 68 cas.

## Sous-ensemble utilisé

La connexion disponible (environ 0,2 à 0,45 Mo/s) ne permettait pas de télécharger les 9,3 Go dans
un délai raisonnable, et l'entraînement se fait sur CPU. J'ai donc procédé ainsi :

1. séparation des 303 cas annotés en 70 % / 15 % / 15 % (212 / 45 / 46 cas) avec la seed 42,
   au niveau du cas (un cas = un patient) ;
2. sélection des premiers cas de chaque ensemble, dont l'ordre est déjà aléatoire :
   40 cas d'entraînement, 8 de validation et 20 de test (68 cas, 1,24 Go) ;
3. téléchargement de ces seuls fichiers par requêtes HTTP partielles. Les fichiers obtenus sont
   identiques octet par octet à ceux de l'archive (même position et même taille) ; leur flux gzip est
   vérifié après téléchargement.

La séparation complète et le sous-ensemble sont enregistrés dans `data/splits/split_seed42.json`.
Le sous-ensemble reste inclus dans la séparation complète : sur GPU, il suffit d'augmenter
`subset_sizes` dans `configs/data.yaml` sans changer l'affectation des cas déjà utilisés.

## Prétraitement

1. Lecture avec NiBabel, réorientation en RAS+ (`as_closest_canonical`, simple permutation ou
   inversion d'axes, sans interpolation), puis stockage en tableau (Z, H, W) dans la convention
   d'affichage radiologique.
2. Vérifications : mêmes dimensions et même matrice affine pour l'image et le masque, labels
   entiers dans {0, 1, 2}.
3. Normalisation : troncature aux percentiles 0,5 et 99,5 des voxels annotés des volumes
   d'entraînement, puis centrage-réduction avec leur moyenne et leur écart-type (schéma CT de
   nnU-Net). Les statistiques sont estimées sur l'ensemble d'entraînement uniquement.
4. Pas de rééchantillonnage dans le plan : toutes les coupes font déjà 512 x 512, et
   l'interpolation des masques (même au plus proche voisin) supprimerait une partie des vaisseaux
   les plus fins, qui sont justement l'objet de l'étude. La variabilité de l'espacement est en partie
   couverte par l'augmentation d'échelle (± 15 %). L'option `target_inplane_spacing` permet de
   rééchantillonner si besoin (interpolation linéaire pour l'image, plus proche voisin pour le
   masque, et retour sur la grille d'origine avant l'évaluation).
5. Stockage en `.npy` (image en float16, masque en uint8) pour une lecture rapide par memory-map.

Les fichiers NIfTI d'origine ne sont jamais modifiés. L'évaluation repart toujours de ces fichiers.
