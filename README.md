# ZayyuNet
A unified deep learning model for the identification of epigenetic modifications using raw genomic sequences

ZayyuNet is a new framework for the prediction of post-transcriptional modifications (6mA/m6A, 4mC, and psi) using raw genomic sequences. The architecture is different from the traditional deep learning models created and inspired by the SpinalNet architecture.

## Abstract
Epigenetic modifications have a vital role in gene expression and are linked to cellular processes such as differentiation, development, and tumorigenesis. Thus, the availability of reliable and accurate methods for identifying and defining these changes facilitates greater insights into the regulatory mechanisms that rely on epigenetic modifications. The current experimental methods provide a genome-wide identification of epigenetic modifications; however, they are expensive and time-consuming. To date, several machine learning methods have been proposed for identifying modifications such as DNA N6-Methyladenine (6mA), RNA N6-Methyladenosine (m6A), DNA N4-methylcytosine (4mC), and RNA pseudouridine ( Ψ ). However, these methods are task-specific computational tools and require different encoding representations of DNA/RNA sequences. In this study, we propose a unified deep learning model, called ZayyuNet, for the identification of various epigenetic modifications. The proposed model is based on an architecture called, SpinalNet, inspired by the human somatosensory system that can efficiently receive large inputs and achieve better performance. The proposed model has been evaluated on various epigenetic modifications such as 6mA, m6A, 4mC, and Ψ and the results achieved outperform current state-of-the-art models. A user-friendly web server has been built and made freely available at http://nsclbio.jbnu.ac.kr/tools/ZayyuNet/ .


The overview of ZayyuNet architecture is illustrated in the figure below:

![block_zayyu](https://user-images.githubusercontent.com/80881943/111714892-01b45700-8896-11eb-8d58-22222aca1c84.png)


**For for details, please refer to the [paper](10.1109/TCBB.2021.3083789)**



