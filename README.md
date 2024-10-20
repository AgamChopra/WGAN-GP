<h1>WGAN-GP</h1>
<p>My code for various GANS related tasks using wgan with gradient penalty. This adversarial model (generator and critic models) were trained on a single NIDIA RTX 3070 8gb GPU. The trained generator requires about 2gb of memory in production(evaluation + no_grad) mode. train.py contains the code for training the generator and critic using wgan-gp. A docker container will be provided in the future if I can get my hands on better hardware and retrain the models with more parameters.</p>

<p align="center">
    <p align="center">
    <a href="https://github.com/AgamChopra/WGAN-GP/raw/main/img/gan_celeb_video.webm" download>
        <img src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/sample_celeb_predictions_14.png" alt="Download Video" width="300" height="300">
    </a>
    <img width="300" height="300" src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/cat_movie_quick.gif">
    <br><i>Fig. Output of generator during traing using celeb face dataset(Left)<i>Click the image to download the WebM video.</i>, using smaller model on cat dataset(Right).</i><br>
    <img width="200" height="200" src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/sample_celeb_predictions_5.png">
    <img width="200" height="200" src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/sample_celeb_predictions_9.png">
    <img width="200" height="200" src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/sample_celeb_predictions_13.png">
    <img width="200" height="200" src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/Figure%202022-06-02%20182811%20(15).png">
    <br><i>Fig. Outputs of trained generators.</i><br>
    <img width="300" height="200"src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/training-celeb.png">
    <img width="300" height="200"src="https://github.com/AgamChopra/WGAN-GP/blob/main/img/training_loss.jpeg">
    <br><i>Fig. Training of celeb and cat generators and critics.</i><br>
</p>

<p><a href="https://raw.githubusercontent.com/AgamChopra/WGAN-GP/main/LICENSE" target="blank">[The MIT License]</a></p>
