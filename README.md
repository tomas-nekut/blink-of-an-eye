## In the wink of an eye

**In the wink of an eye** is a Google Chrome extension that makes images of Czech president Miloš Zeman more realistic. If you browse the Internet with this extension installed and activated, every face of Zeman appearing in any image on any website will be automatically animated so that he will wink and smile at you. Isn’t it great?

This is the semestral project I worked on within the _Open Source Software_ course at _Seoul National University of Science and Technology_ under professor _Sunglok Choi, Ph.D_. The source code of this project includes the extension itself and a web application written in Python responsible for all image-processing.

IMAGE

---

### The motivation

**Why Miloš Zeman and winking?** Because of this awkward moment when the Czech president, believe it or not, winked at a reporter in the middle of an interview on television. This happened during a debate before the Czech presidential election in 2018, but even so, he made it to the presidency. Anyway, you can watch the famous moment [here](https://bit.ly/3WurLlX).

**Why Miloš Zeman?** Because, to the best of my knowledge, there isn’t any other president who would wink at a reporter.

**Why?** Because why not… :wink:

---

### Get started

There are two simple steps to make Zeman wink at you.

1.  Set up the image-processing web server. Navigate to the `/server` directory where you will find a Python web application based on the _FastAPI_ framework. You can run in using `python server.py --port <port>`. It should run on a public domain to allow the browser extension to reach its endpoint. TIP! You can use [localtunnel](https://theboroer.github.io/localtunnel-www/) to make your local server publically available. If you are a fan of **Google Collaboratory,** you can simply launch the server using a `/server/server.ipynb` notebook using their free GPU resources.
2.  Install the Chrome extension. You can find the source code in the `/chrome_extension` directory. There is a [tutorial](https://developer.chrome.com/docs/extensions/mv3/getstarted/development-basics/#load-unpacked) on how to load an unpacked extension to your browser. To make everything work together, you have to update the URL of your image-processing server in the very beginning of `/chrome_extension/script.js`.

---

### How does it work?

---

### License

---

### Conclusion

This project mainly has an entertaining purpose. But I choose this topic also because I wanted to make any visitor to this repository, just like you, who reads it to the very end, think a little about one important problem. In this case, you can clearly recognize that the animated Zeman is fake, since the goal of this project obviously wasn't to produce perfectly realistic images. But what if the result was perfectly realistic, even realistic enough that you wouldn't be able to tell it was fake anymore? State-of-the-art deep learning models are already pretty close to that point. Imagine it was possible to make a video of a person doing or saying anything you wanted, especially something the person would never do or say. How would you then know what was real and what wasn't? Quite scary, right? Does it mean that these so-called **deep-fake** systems are bad? Not really, because…

“A sword is never a killer, it is a tool in the killer's hands.”    — _Lucius Annaeus Seneca_.

Think about it, because the moment is coming. It is actually coming so fast that it will become reality **in the wink of an eye**. Does the name of this project finally make sense? :wink: