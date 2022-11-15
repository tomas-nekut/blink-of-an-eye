## In the wink of an eye

**In the wink of an eye** is an extension for Google Chrome that makes images of Czech president Miloš Zeman more realistic. If you browse the Internet with this extension installed and activated, every face of Zeman appearing in any image on any website will be automatically animated so that he will wink and smile at you. Isn’t it great?

This is the semestral project I worked on within the _Open Source Software_ course at _Seoul National University of Science and Technology_ under professor _Sunglok Choi, Ph.D._. The source code of this project includes the extension itself and a web application written in Python responsible for all image-processing.

![](https://plus.rozhlas.cz/sites/default/files/styles/cro_16x9_tablet/public/images/b511ee7b9276eb0d403949cfca7abdc1.jpg) ![](examples/1.png)

---

### The motivation

**Why winking Miloš Zeman?** Because of this awkward moment when the Czech president, believe it or not, winked at a reporter in the middle of an interview on television. This happened during a debate before the Czech presidential election in 2018, but even so, he made it to the presidency. Anyway, you can watch the famous moment [here](https://bit.ly/3WurLlX).

---

### Get started

There are two simple steps to make Zeman wink at you.

1.  Set up the image-processing web server. Navigate to the `/image_processing` directory where you will find a Python web application based on the _FastAPI_ framework. You can run in using `python server.py --port <port> [--face_example <face_example_path>] [--motion_vectors <motion_vectors_path]`. See _It does not have to be only a winking Zeman_ chapter to see how to use optional parameters. The server should run on a public domain to allow the browser extension to reach its endpoint. TIP! You can use [localtunnel](https://theboroer.github.io/localtunnel-www/) to make your local server publicly available. If you are a fan of **Google Collaboratory,** you can simply launch the server using a `/image_processing/colab_server.ipynb` notebook and use free GPU resources.
2.  Install the Chrome extension. You can find the source code in the `/chrome_extension` directory. There is a [tutorial](https://developer.chrome.com/docs/extensions/mv3/getstarted/development-basics/#load-unpacked) on how to load an unpacked extension to your browser. To make everything work together, you have to update the URL of your image-processing server at the very beginning of `/chrome_extension/script.js`.

---

### How does it work?

The image-processing server does nothing but transfers a movement from a video to an arbitrary static image of a face. This results in an animated face mimicking the same facial expression as the actor did in the source video. This transformation is restricted only to Miloš Zeman's face.

The image-processing stands on 3 main components: [**Face Recognition**](https://github.com/ageitgey/face_recognition), [**MediaPipe Face Mesh**](https://google.github.io/mediapipe/) and [**OpenCV**](https://docs.opencv.org/3.4/index.html).

[**Face Recognition**](https://github.com/ageitgey/face_recognition) is an open-source Python library that solves, besides other tasks, also _face detection_ and _face identification_ problems. This library is used to detect Zeman's face. 

[**MediaPipe**](https://google.github.io/mediapipe/) is an open-source machine learning framework by Google which offers a solution to common computer vision tasks. I used specifically the [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) tool which estimates the positions of 468 facial points in the 3D space given a photo of a face.

![MediaPipe Face Mesh](https://raw.githubusercontent.com/patlevin/face-detection-tflite/main/docs/portrait_fl.jpg)

I used this library to preprocess the source video by detecting facial points in each frame and computing motion-vectors representing the motion in the 3D space of each facial point in each frame. This motion can then be applied to a new face.

[**OpenCV**](https://docs.opencv.org/3.4/index.html) is an open-source computer vision library. This project uses OpenCV mainly to transform faces i.e. apply motion-vectors to its facial points. This task can be solved using [Remap](https://docs.opencv.org/3.4/d1/da0/tutorial_remap.html). This function maps pixels from their original position to another and thus performs any transformation of the input image. It requires defining a mapping function a.k.a. where each pixel should move. Since the motion-vectors obtained from the source video define this mapping function only in facial points, these vectors have to be interpolated to each pixel. For this purpose [LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html) from _Scipy_ library worked the best.

Using the tools described above I was able to transfer a facial expression from a video to a static image. Sounds simple… But as always there were several catches. For example, if two faces are _not oriented the same way_ it is not possible to transfer any motion vector from one to another. What solved this problem was normalizing facial points first. So the procedure is as follows: rotate, scale, and translate facial points in 3D space to a common normalized front-facing pose then add motion-vectors and finally apply an inverse transformation to denormalize them. Another problem was _teeth_ which have to show up when the face is smiling. Teeth have to reflect the pose of a face and light conditions as well. You are free to see the source code to see more detail.

---

### It does not have to be only a winking Zeman

If you would like to experiment you can animate a different person instead of Zeman. The only thing you need to do is specify a path to your face example image of whoever’s face you want to animate by using `--face_example` argument of `server.py` script.

You can even change the facial expression by providing a path to motion-vectors file using `--motion_vectors` parameter. To extract motion vectors from your video, use the `/image_processing/motion_vectors_extraction.py`. You can run the script like this `python motion_vectors_extraction.py --dst <dst_path> --videos <video_path_1> [<weight_1>] [<video_path_2> [<weight_2>] ...]`. You can specify as many videos as you want. Motion vectors will be extracted from each of them and added together weighted by a given weight resulting in a blend of all expressions from all videos. The default weight is `1`.

Motion-vectors file `/image_processing/assests/wink.npy` was made out of 2 videos `/image_processing/assests/videos/1.mp4` (the wink) weighted by `1.2` and `/image_processing/assests/videos/2.mp4` (the smile) weighted by `0.8`. These are a scene from the _Oru Adaar Love_ movie and a part of an educational video on facial expressions by [_Management Development International_](https://www.mdi-training.com/) respectively. 

---

### License

This project is licensed under the terms of the GPLv3 license.

---

### Conclusion

This project mainly has an entertaining purpose. But I choose this topic also because I wanted to make any visitor to this repository, just like you, who reads it to the very end, think a little about one important problem. In this case, you can clearly recognize that the animated Zeman is fake since the goal of this project wasn't to produce perfectly realistic images. But what if the result was perfectly realistic, even realistic enough that you wouldn't be able to tell it was fake anymore? State-of-the-art deep learning models are already pretty close to that point. Imagine it was possible to make a video of a person doing or saying anything you wanted, especially something the person would never do or say. How would you then know what was real and what wasn't? Quite scary, right? Does it mean that these so-called **deep-fake** systems are bad? Not really, because…

“A sword is never a killer, it is a tool in the killer's hands.”    — _Lucius Annaeus Seneca_.

Think about it, because the moment is coming. It is coming so fast that it will become reality **in the wink of an eye**. Does the name of this project finally make sense? :wink: