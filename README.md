Try_POF
===
discription
---
this project try to achieve part of the method discribed in [1812.01598](https://arxiv.org/pdf/1812.01598.pdf)

the method in paper chould detect 3d poses from a single RGB image and generate 3d model,with a group of images croped from a video, the paper come out with a way to adjust models to make them seems coherent,their project page is [here](http://domedb.perception.cs.cmu.edu/mtc.html)

the method in paper can be divided into 3 parts:

part 1:<br>
get 3d joints position(in (x,y,z) format) from image.<br>
the paper come out POF for this part,it looks like the PAF in [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) but it's 3d vectors format.<br>
this project mainly try to achieve this part(just focus on body pose,since hand and face is very similar)<br>
<br>
part 2:<br>
convert the joints information into mesh models.<br>
this part the paper include [Adam](https://arxiv.org/pdf/1801.01615.pdf) model<br>
<br>
part 3:<br>
adjust models to make them linked together<br>
