
## Bag Of Words For Image Recognition

In computer science, there is a lot of method for image recognation. Nowadays, deep learning very
populer. Image recognation is a hard problem but we have a diffirent method which name is Bag Of
Visual Words. Its different other methods because it has different story. It came another area. The bag-
of-words model is a simplifying representation used in natural language processing and information
retrieval.

You have to setup libsvm library.

What does this code do?  
1.Extract SURF descriptors from all training folders.  
2.Clustering this descriptors with k-means for Vocabulary.  
3.Extract SURF descriptor each image in train folder and convert to BoW  
4.Train a SVM classifier data which came step 3.  
5.Extract SURF descriptor each image in test folder and convert to BoW  
6.Test classifier with test images fatas came from 5

You can find detailed report in report folder.
