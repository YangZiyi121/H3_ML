<h3><center>CS229: Homework 3</center></h3>

<center>Ziyi Yang</center>

<h4>Evironment Settings</h4>

In this project, the setting of the VM I used is shown below

|         Specs         |    OS/SW     | Hosted Hardware |
| :-------------------: | :----------: | :-------------: |
| RyzenTR Pro 32C, 512T | Ubuntu 18.04 |     RTX6000     |

<h4>Test Results</h4>

For all the experiments, the size of **training set** is 54000,  **validation set** is 6000, **testing set** is 10000

* Number of hidden node = 4

| n_hidden | lambda | Training set accuracy | Validation set accuracy | Test set accuracy | Time cosumed for one run |
| :------: | :----: | :-------------------: | :---------------------: | :---------------: | :----------------------: |
|    4     |   0    |        65.03%         |         66.51%          |      65.71%       |         20.588s          |
|    4     |  0.2   |        48.58%         |         47.82%          |      47.92%       |         22.253s          |
|    4     |  0.4   |        61.04%         |         61.67%          |      60.74%       |         24.422s          |
|    4     |  0.6   |        72.07%         |         72.15%          |      71.93%       |         21.492s          |
|    4     |  0.8   |        74.68%         |         74.61%          |       75.5%       |         19.786s          |
|    4     |   1    |        58.53%         |         58.48%          |      57.97%       |         20.662s          |

* Number of hidden node = 8

| n_hidden | lambda | Training set accuracy | Validation set accuracy | Test set accuracy | Time cosumed for one run |
| :------: | :----: | :-------------------: | :---------------------: | :---------------: | :----------------------: |
|    8     |   0    |        89.52%         |         89.47%          |      89.64%       |         23.259s          |
|    8     |  0.2   |        88.54%         |         87.63%          |      88.61%       |          23.66s          |
|    8     |  0.4   |        81.62%         |         81.83%          |      81.75%       |         21.933s          |
|    8     |  0.6   |        88.77%         |         88.26%          |      88.52%       |         22.725s          |
|    8     |  0.8   |        82.67%         |         82.18%          |      82.56%       |         22.784s          |
|    8     |   1    |        88.91%         |          88.0%          |      88.81%       |         24.437s          |

* Number of hidden node = 12

| n_hidden | lambda | Training set accuracy | Validation set accuracy | Test set accuracy | Time cosumed for one run |
| :------: | :----: | :-------------------: | :---------------------: | :---------------: | :----------------------: |
|    12    |   0    |        91.77%         |         91.53%          |      91.67%       |         26.032s          |
|    12    |  0.2   |        91.73%         |         90.81%          |      91.28%       |         25.833s          |
|    12    |  0.4   |        91.55%         |          91.1%          |      91.66%       |         25.533s          |
|    12    |  0.6   |        90.62%         |         90.13%          |      90.56%       |         25.854s          |
|    12    |  0.8   |        90.34%         |         90.22%          |      90.41%       |         25.324s          |
|    12    |   1    |        91.57%         |         90.78%          |      91.39%       |         26.151s          |

* Number of hidden node = 16

| n_hidden | lambda | Training set accuracy | Validation set accuracy | Test set accuracy | Time cosumed for one run |
| :------: | :----: | :-------------------: | :---------------------: | :---------------: | :----------------------: |
|    16    |   0    |        92.81%         |          92.6%          |      92.25%       |         27.522s          |
|    16    |  0.2   |        91.27%         |         90.33%          |      90.73%       |         26.929s          |
|    16    |  0.4   |        92.89%         |          92.5%          |      92.53%       |         27.337s          |
|    16    |  0.6   |        93.17%         |         92.28%          |      92.72%       |         27.114s          |
|    16    |  0.8   |        92.29%         |         92.33%          |      92.06%       |         26.986s          |
|    16    |   1    |        93.07%         |          92.6%          |      92.57%       |         28.714s          |

* Number of hidden node = 20

| n_hidden | lambda | Training set accuracy | Validation set accuracy | Test set accuracy | Time cosumed for one run |
| :------: | :----: | :-------------------: | :---------------------: | :---------------: | :----------------------: |
|    20    |   0    |        93.83%         |         93.05%          |      93.51%       |         33.738s          |
|    20    |  0.2   |        93.74%         |         92.75%          |      93.37%       |         33.597s          |
|    20    |  0.4   |        91.88%         |         91.27%          |      91.52%       |         30.821s          |
|    20    |  0.6   |        93.25%         |         92.98%          |      93.06%       |         30.961s          |
|    20    |  0.8   |        93.32%         |         92.52%          |      92.99%       |         30.369s          |
|    20    |   1    |        93.73%         |         93.03%          |      93.47%       |         34.685s          |

