# Amazon Web Services Cluster Start-Up
In this file we will detail how we set up an Amazon Web Services (AWS) message passing interface (MPI) cluster used to conduct experiments. 

Our cluster was built by using nine m4.2xlarge instances. More details about the specs for this machine can be found the [project website](https://rednotion.github.io/parallel_esn_web/) or from [AWS](https://aws.amazon.com/ec2/instance-types/). The core instructions used to connect the nodes can be found in [CS205 MPI Infrastructure Guide](https://docs.google.com/document/d/1Dw9vUSqOClWS26ypyYGniksdwPmqcSSFPQBafYFlyMs/edit). The guide provides the exact steps needed to build a two node MPI cluster on AWS. 

In this project we generalize the guide because we worked with a nine node system. To accomplish this we followed the instructions and in particular whenever the guide stated to take an action for node one, we repeated it for the remaining seven machines. Moreover when enabling a password-free SSH between the instances, one must pass the public RSA key to the other nodes and repeat this for each of the instances in the cluster. Other than those two points, if one follows the previously listed guide, there should not be any issues setting up the cluster we used for our experiments.