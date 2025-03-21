We process one-dimensional data sequentially into RGB image data.  For each dataset, there are both attack messages and normal messages. If an image is composed entirely of normal messages, we will label it as normal image. Otherwise, we will label it as attack image. Therefore, each dataset can generate two sets of image. 

See https://github.com/QiguangJiang/StatGraph/tree/main/BaselineModels/CarHacking and https://github.com/QiguangJiang/StatGraph/tree/main/BaselineModels/ROAD for the specific code of data preprocessing.

The key code for each baseline method is "train-CHD", "train-ROAD", "test-CHD" and "test-ROAD".