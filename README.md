# RamanSpectrumEffect
Raman spectroscopy is widely used for its ability to provide rich molecular-level information
from biological samples. However, the high sensitivity of Raman spectra can also unintentionally encode
extraneous information, such as patient identity, instrument-specific signatures, environmental conditions,
and sample handling protocols. This hidden structure, referred to as the batch effect, poses increasing
challenges in machine learning applications, where it can lead to model overfitting, poor generalization,
and even privacy risks when datasets are shared. In this work, we investigate the presence and impact of
the batch effect in a Raman spectroscopy dataset derived from glioma tumor samples. We show that deep
learning can accurately predict patient identity from individual spectra, revealing strong bias patterns in the
data. To mitigate this issue, we explore several approaches and introduce an algorithmic framework based
on adversarial deep learning. Our method effectively reduces the influence of confounding sample-specific
signals while preserving information relevant to tumor classification. These findings highlight the need for
careful bias control in Raman-based AI pipelines and provide tools to support more robust and privacy-aware
applications in biomedical spectroscopy.

Data: https://seafile.utu.fi/d/dbaf4b8e81b5436da03a/

### Keywords: batch effect, deep learning, feature importance, glioma, Raman spectroscopy

![Feature learning]([https://github.com/](https://github.com/JoelSjoberg/RamanSpectrumEffect/blob/main/(MANUAL)FeatureImportanceEvolution_0.gif))
