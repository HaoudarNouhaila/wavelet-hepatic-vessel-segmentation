# Références

## Données

- Simpson, A. L., Antonelli, M., Bakas, S., et al. (2019). *A large annotated medical image dataset
  for the development and evaluation of segmentation algorithms*. arXiv:1902.09063.
  Référence demandée par le site du Medical Segmentation Decathlon pour citer les données.
- Antonelli, M., Reinke, A., Bakas, S., et al. (2022). *The Medical Segmentation Decathlon*.
  Nature Communications, 13, 4128.
- Medical Segmentation Decathlon, page officielle : http://medicaldecathlon.com/ ; téléchargement
  AWS : http://medicaldecathlon.com/dataaws/. Données sous licence CC-BY-SA 4.0 (Task08 : Memorial
  Sloan Kettering Cancer Center).

## Méthodes

- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical
  Image Segmentation*. MICCAI 2015, LNCS 9351, p. 234-241. Architecture de base des modèles 2D.
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., Ronneberger, O. (2016). *3D U-Net: Learning
  Dense Volumetric Segmentation from Sparse Annotation*. MICCAI 2016. Extension volumique.
- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., Maier-Hein, K. H. (2021). *nnU-Net: a
  self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods,
  18, 203-211. Schéma de normalisation CT, perte Dice + cross-entropy, Dice calculé sur le lot.
- Milletari, F., Navab, N., Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for
  Volumetric Medical Image Segmentation*. 3DV 2016. Dice loss.
- Shit, S., Paetzold, J. C., Sekuboyina, A., et al. (2021). *clDice - a Novel Topology-Preserving
  Loss Function for Tubular Structure Segmentation*. CVPR 2021. Métrique clDice.
- Lee, T. C., Kashyap, R. L., Chu, C. N. (1994). *Building skeleton models via 3-D medial
  surface/axis thinning algorithms*. CVGIP: Graphical Models and Image Processing, 56(6), 462-478.
  Squelettisation 3D utilisée par scikit-image.
- Mallat, S. G. (1989). *A theory for multiresolution signal decomposition: the wavelet
  representation*. IEEE TPAMI, 11(7), 674-693. Transformée en ondelettes discrète 2D.
- Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM. Ondelettes de Daubechies.
- Nason, G. P., Silverman, B. W. (1995). *The Stationary Wavelet Transform and some Statistical
  Applications*. Lecture Notes in Statistics, 103. Transformée stationnaire (non décimée).
- Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., Eubank, W. (2003). *PET-CT image
  registration in the chest using free-form deformations*. IEEE TMI, 22(1), 120-128. Information
  mutuelle de Mattes (extension recalage).
- Fitzpatrick, J. M., West, J. B. (2001). *The distribution of target registration error in
  rigid-body point-based registration*. IEEE TMI, 20(9), 917-927. Erreur de recalage sur cibles (TRE).

## Logiciels

- PyTorch (Paszke et al., 2019, NeurIPS) ; MONAI (Cardoso et al., 2022, arXiv:2211.02701) ;
  PyWavelets (Lee et al., 2019, Journal of Open Source Software, 4(36), 1237) ; NiBabel ;
  SimpleITK (Lowekamp et al., 2013, Frontiers in Neuroinformatics ; Yaniv et al., 2018, Journal of
  Digital Imaging) ; scikit-image (van der Walt et al., 2014, PeerJ) ; SciPy ; NumPy ; pandas ;
  scikit-learn ; Matplotlib ; seaborn.

## Projet de départ

Avant ce projet, j'avais reproduit un projet de segmentation cardiaque CT par U-Net (TensorFlow /
Keras) publié en quatre dépôts par l'utilisateur GitHub AhmedIbrahimai, avec le jeu de données
Kaggle « CT Heart Segmentation » (nikhilroxtomar) :

- https://github.com/AhmedIbrahimai/CT-Heart-segmentation-using-U-NET-part-one (préparation)
- https://github.com/AhmedIbrahimai/CT-Heart-segmentation-using-U-NET-part-two-training-the-model
- https://github.com/AhmedIbrahimai/CT-Heart-segmentation-using-U-NET-part-three-evaluate-the-model
- https://github.com/AhmedIbrahimai/CT-Heart-segmentation-using-U-NET-part-three-test-the-model
- https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation

Ces dépôts n'indiquent pas de licence : je les ai utilisés uniquement comme référence pour les étapes
générales (préparation, entraînement, évaluation, test) et aucun code n'en a été copié. Le présent
projet est écrit en PyTorch et suit une organisation différente. Les points que j'ai repris ou
modifiés sont décrits dans `docs/methodologie.md`.
