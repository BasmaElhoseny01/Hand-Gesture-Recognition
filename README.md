
<h1 align="center">Hand Gesture Recognition 🤙</h1>

<p>Given an image containing a single hand 🖐, our system classifies the hand gesture into one of six digits <em>from 0️⃣ to 5️⃣</em>. It handles different lighting effects⚡ and different hand poses as much as possible.</p>
<p>A complete machine learning pipeline, composed of the following modules:</p>
<ul>
  <li>Preprocessing</li>
  <li>Feature extraction</li>
  <li>Classification</li>
</ul>
<p>Wait for the classification stage to see our magical touch ✨</p>

<br>

## Preprocessing
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

## Feature Extraction
<p>This module aims to describe the image with a constant size feature vector. We tried various options to achieve this. Here they are:</p>
<ul>
  <li>Using OCR extracted from the previous stage</li>
  <li>Histogram of Oriented Gradients</li>
  <li>SIFT (failed)</li>
  <li>ORB (failed)</li>

</ul>

<br>

## Classification
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
<p>The model ouputs the HOG_SVM result in case a tie happens</p>

<br>

## How to run our model on your test set ##

```bash
# Clone this project
$ git clone https://github.com/{{YOUR_GITHUB_USERNAME}}/hand-gesture-recognition

# Access
$ cd Hand-Gesture-Recognition

# Install dependencies
$ pip install -r requirements.txt

# Go to the run directory
$ cd .\src\final
```
### Put the expected.txt in the same directory
### Put your images in the 'data' folder
#### It's better to label your images from 1 to n

```bash
# Go back to the root directory
$ cd ..\..\
# Run the following batch script
$ .\src\final\run.bat

# The results.txt and time.txt will be generated in the same directory
# The script will output the accuracy as well
```

<!-- Contributors -->
## Contributors ✨

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

## Feedback  

If you have any feedback, please reach out to us at mohabmohamedmohamedzaghloul@gmail.com


