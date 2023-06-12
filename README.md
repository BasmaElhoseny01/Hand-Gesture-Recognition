<h1 align="center">Hand Gesture Recognition 🤙</h1>



## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#Overview"> 📙 Overview</a>
- <a href ="#Achievement"> ✨ Our Achievement</a>
- <a href ="#started">  🚀 Get Started</a>
- <a href ="#modules"> 📦 Modules</a>
  - <a href ="#preprocessing"> 📸 Preprocessing</a>
  - <a href ="#feature"> ⛏ Feature Extraction</a>
  - <a href ="#classification"> ❓ Classification</a>
- <a href ="#contributors"> 🧑 Contributors</a>
- <a href ="#feedback"> 😉 FeedBack</a>
- <a href ="#license"> 🔒 License</a>

<br>

## <img align="center"  width =50px  height =50px src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/waving-hand_1f44b.gif"> Overview <a id = "Overview"></a>
Given an image containing a single hand 🖐, our system classifies the hand gesture into one of six digits <em>from 0️⃣ to 5️⃣</em>. It handles different lighting effects⚡ and different hand poses as much as possible.


## <img align="center"  width =60px  height =70px src="https://opengameart.org/sites/default/files/gif_3.gif"> Our Achievement <a id = "Achievement"></a>
We  have been ranked as the **3rd team** on the leader-board out of 20 teams with an accuracy of **73%** on the hidden test set 


## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>
## How to run our model on your test set 🤔🤔 ##
```bash
# Clone this project
$ git clone https://github.com/BasmaElhoseny01/Hand-Gesture-Recognition

# Access
$ cd Hand-Gesture-Recognition

# Install dependencies
$ pip install -r requirements.txt

# Go to the run directory
$ cd .\src\test
```
### Put the expected.txt in the same directory
### Put your images in the 'data' folder
``` It's better to label your images from 1 to n ```

```bash
# Go back to the root directory
$ cd ..\..\
# Run the following batch script
$ .\src\final\run.bat

# The results.txt and time.txt will be generated in the same directory
# The script will output the accuracy as well
```



## <img align="center"  width =60px  height =70px src="https://media0.giphy.com/media/dAoHbGjH7k5ZTeQeBI/giphy.gif?cid=6c09b952bc7553e890ee5e534c14fb9f7d081b4676991d9b&ep=v1_internal_gifs_gifId&rid=giphy.gif&ct=s"> Modules <a id = "modules"></a>
<p>A complete machine learning pipeline, composed of the following modules:</p>
<ul>
  <li>Preprocessing</li>
  <li>Feature extraction</li>
  <li>Classification</li>
</ul>
<p>Wait for the classification stage to see our magical touch ✨</p>

[![Total Commits](https://img.shields.io/github/commit-activity/y/BasmaElhoseny01/Hand-Gesture-Recognition?style=flat-square)](https://github.com/BasmaElhoseny01/Hand-Gesture-Recognition)  
[![GitHub stars](https://img.shields.io/github/stars/BasmaElhoseny01/Hand-Gesture-Recognition?style=flat-square)](https://github.com/BasmaElhoseny01/Hand-Gesture-Recognition)

<br>



## <img align="center"  width =60px  height =70px src="https://media3.giphy.com/media/psneItdLMpWy36ejfA/source.gif"> Preprocessing <a id = "preprocessing"></a>

<p>This module aims to handle the variation in the images color, texture, size, orientation and lighting conditions. It bridges the gap between different hands of same gesture by applying the following steps:</p>
<ul>
  <li>Resize the image for faster processing</li>
  <li>Convert the image to HSV</li>
  <li>Extract S and V channels</li>
  <li>Get the thresholds of hand and apply it in both channels</li>
  <li>Bitwise AND the two thresholded images</li>
  <li>Filling the empty regions</li>
  <li>Resize region filled mask to be compatable with original image size</li>
  <li>Remove padding</li>
  <li>Horizontally flip the image, if required</li>
  <li>Extract OCR of the image from the previous step</li>
</ul>
<p><em>Note:</em> We tried about 🔟 other methods. This was the most robust one🤖</p>

<br>


## <img align="center"  width =60px  height =70px src="https://media4.giphy.com/media/ux6vPam8BubuCxbW20/giphy.gif?cid=6c09b952gi267xsujaqufpqwuzeqhbi88q0ohj83jwv6dpls&ep=v1_stickers_related&rid=giphy.gif&ct=s"> Feature Extraction <a id = "feature"></a>

<p>This module aims to describe the image with a constant size feature vector. We tried various options to achieve this. Here they are:</p>
<ul>
  <li>Using OCR extracted from the previous stage</li>
  <li>Histogram of Oriented Gradients</li>
  <li>SIFT (failed)</li>
  <li>ORB (failed)</li>

</ul>

<br>


## <img align="center"  width =60px  height =70px src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Thinking_-_Idil_Keysan_-_Wikimedia_Giphy_stickers_2019.gif"> Classification <a id = "classification"></a>

<p>This module includes our trained model, or should we say our trained <em>models</em></p>
<p>We mainly depend on two classical ML algorithms [Support Vector Machines - Random Forest]</p>
<ul>
  <li>Loads 3 different models
    <ul>
      <li>HOG descriptor, SVM</li>
      <li>HOG descriptor, RF</li>
      <li>OCR descriptor, SVM</li>
    </ul>
  </li>
  <li>Runs each of them on a single thread</li>
  <li>Collects votes from them</li>
  <li>Classifies the image according to the majority of votes</li>
</ul>
<p>The model chose the HOG_SVM result in case a tie happens</p>

<br>


<!-- Contributors -->
## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/BasmaElhoseny01"><img src="https://avatars.githubusercontent.com/u/72309546?v=4" width="150px;" alt=""/><br /><sub><b>Basma Elhoseny</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/yasmineghanem"><img src="https://avatars.githubusercontent.com/u/74925701?v=4" width="150px;" alt=""/><br /><sub><b>Yasmine Ghanem</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Mohabz-911" ><img src="https://avatars.githubusercontent.com/u/68201932?v=4" width="150px;" alt=""/><br /><sub><b>Mohab Zaghloul</b></sub></a><br />
    </td>
     <td align="center"><a href="https://github.com/YasminElgendi"><img src="https://avatars.githubusercontent.com/u/54359829?v=4" width="150px;" alt=""/><br /><sub><b>Yasmin Elgendi</b></sub></a><br /></td>
  </tr>
</table>

<br>

## <img align="center"  width =50px  height =50px src="https://images.squarespace-cdn.com/content/v1/5c88c50af4e5316a44e9f34e/1639666090540-WIW96612QF3IQPGQXPD3/giphy+%284%29.gif"> FeedBack <a id = "feedback"></a>
> If you have any feedback, please reach out to us at mohabmohamedmohamedzaghloul@gmail.com.

<br>


## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/ggoKD4cFbqd4nyugH2/giphy.gif?cid=6c09b9527jpi8kfxsj6eswuvb7ay2p0rgv57b7wg0jkihhhv&rid=giphy.gif&ct=s"> License <a id = "license"></a>
This software is licensed under MIT License, See [License](https://github.com/BasmaElhoseny01/Hand-Gesture-Recognition/blob/main/LICENSE) for more information ©Basma Elhoseny.