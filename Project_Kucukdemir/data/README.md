The original paper uses data from the following datasets:
* [MPI Dyna](http://dyna.is.tue.mpg.de/)
* [SMPL](https://smpl.is.tue.mpg.de/)

Another option would be the SCAPE dataset, which is available [here](https://ai.stanford.edu/~drago/Projects/scape/scape.html). However, before using the data for the method, you have to preprocess it so that you can get feature vectors for each vertex (as described in Section 3.1). For this purpose [ACAP_Linux](https://github.com/gaolinorange/Automatic-Unpaired-Shape-Deformation-Transfer/tree/master/ACAP_linux), which is implemented and shared by Gao et al., can be used. After computing feature vectors, you have to store them in a '.h5' file, which has the keys defined in "__getitem__" method of "SMDSwBdLSTMDataset" class. See [main.py](https://github.com/CENG502-Projects/CENG502-Spring2022/blob/ff-k/Project_Kucukdemir/code/main.py) for the details.

If you consider using another dataset, please note that all meshes in that dataset should be of equal size in terms of number of vertices.
