GaussianSplattingViewer-Builtwith-a-Single-Python-File
======

Place this file under the 3dgs directory to use it. 

I created this file primarily for quickly viewing training results. This visualization file achieves visualization by calling some functions from 3dgs, so it can be easily modified and used in many projects. Additionally, the file is very simple, allowing for convenient addition of desired visualization features.
![image](https://github.com/user-attachments/assets/b79ac64b-ba91-4ff6-97f0-ce8710046b5c)


Usage
======
I am using it on a Windows system, so it may have issues on other systems.

In addition to the environment required by 3dgs, you also need to install OpenCV
```python
pip install opencv-python
```

Change the checkpoint in the file to the path of the PLY file you want to visualize.
```python
checkpoint = "your address" 
```

Launch viewer
```python
python Visual_Gaussian.py
```

Use W, A, S, D, Q, E to control camera movement, and use J, K, L, I, U, O to control camera view movement.

Use - and = to control the size of the Gaussian kernel.

Use C to exit the screen.


