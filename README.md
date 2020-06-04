# TikTok Trendy Music & Lyrics Generation

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members:
- Jou-Ying Lee, jol067@ucsd.edu
- Yunlin Tang, yut085@ucsd.edu
- Yupei Zhou, yuz522@ucsd.edu
- Sizhu Chen, sic100@ucsd.edu
- Yuanbo Shi, yus263@ucsd.edu

## Abstract

(10 points)

All sharing a Chinese cultural background, our group decides to base this generative art project primarily on music production of popular Chinese song tracks.
<p align="center">
<img width=500px src="https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/TikTok_logo.svg/1920px-TikTok_logo.svg.png">
</p>
TikTok is a video-sharing social networking service owned by ByteDance, that has become one of the most downloaded mobile apps of the decade. With increasingly emerging markets all around the world, musics created, played, and sang on TikTok trend with great popularities, and pieces that top the charts of "most played songs on TikTok" has become a commonly referenced benchmark in China for signs of "trendy music." With this idea in mind, our group hopes to study these popular songs by decomposing the music audios and recomposing them into new audio tracks. Additionally, we are looking to employ text generating algorithms for novel lyric compositions based off of these trained music tracks. <br><br>

Specifically, we would be looking at current Chinese songs topping [TikTok Most-Played Music Chart](https://y.qq.com/n/yqq/toplist/60.html). Also, for generative modelling purposes, we would thereby only focus on one single instrument playing in the background music -- picked to be piano for this particular topic of interest.<br><br>
Building upon text generation with RNN introduced in class, we will be practicing the techniques for processing, predictive, and modelling goals on our own dataset of Chinese characters -- for lyric production purposes. Additionally, we will extend practices introduced on Google Magenta by combining with the use of the Spleeter library for music generations. From here, we are also aware of the many challenges involved: these include unpredictable performances and outputs our model has on Mandarin applications, processing inadaquacy in accurately recognizing input notes therefore leading to biased results, and possible failures in generating alternative note sequences while keeping the styles or harmonies, etc. However, we are also hoping to embrace all of these difficulties and consider them all as an exploratory part of our project. <br><br>
Judging from our prior understandings of most-played TikTok musics, we are anticipating the generated music piece to have a "poppy" or "bouncy" melody similar to most of those on the current chart. Although we do not expect the generated lyrics to make perfect sense, we do anticipate the overall content to convey of a "lovely theme" -- as this is again, what we are seeing from the latest songs topping the chart.<br><br>
Our project results would then be presented as a video clip of a sang out piece of our generated lyrics together with the generated audio piece as the background music. Lyrics are to be sang in Chinese, therefore we will also be providing a running display in the video of the exact lyrics (in Mandarin Chinese with English translations).<br><br>
Ideas and concepts of this work references a few prior projects and papers listed as follows:

- [MidiMe: Personalizing a MusicVAE model with user data](https://research.google/pubs/pub48628/)
- [Generation of music pieces using machine learning: long short-term memory neural networks approach](https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972)
- [Neural Nets for Generating Music](https://medium.com/artists-and-machine-intelligence/neural-nets-for-generating-music-f46dffac21c0)
- [Classical Piano Composer](https://github.com/Skuldur/Classical-Piano-Composer)
- [Play it again, AI: a look at Google Magenta and Machine Learning for audio](https://tech.uqido.com/2020/02/13/play-it-again-ai-a-look-at-google-magenta-and-machine-learning-for-audio/)

## Data and Model

(10 points)

In the final submission, this section will describe both the data you use for this project and any pre-existing models/neural nets. For each you should provide the name, a textual description, and a link. If there is a paper (for neural net) link that as well.
- Such and such Neural Net. The short description of this neural net.
  - [link to code]().
  - [Title of Paper with Link]().
- Training data. Short description of training data including bibliographic info. [link to data]().

#### Model
Our project can be broken down to two major parts, that is audio generation and text generation. With different goals of applications, each are implemented with different models as well. <br>
1. Audio Generation
	- Classical-Piano-Composer:<br>

	- MelodyRNN:<br>
This model is made available from `magenta.music`. It is an LSTM-based language model for musical notes, and is best at continuing a NoteSequence given to it.<br>
For our particular implemented model, we leveraged pre-trained [basic_rnn](http://download.magenta.tensorflow.org/models/basic_rnn.mag) (.mag bundle files) supplied by magenta, and is then trained on the outcome of that Classical-Piano-Composer produced above. <br>
Details regarding this model is linked [here](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn).

2. Text-Generation
	- model goes here<br>

#### Data
- All data used in this project are personally obtained. For our generative tasks are seperated into audio and text production, raw data trained also include each seperate audio files and text files.<br>
- Our audio datas are originally videos obtained from [Youtube](https://www.youtube.com/) and [bilibili](https://www.bilibili.com/). 
- [audio_midi.py](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/audio_midi.py) contains all videos acquired. This python file is then imported into ![audio-to-midi] for processing and conversions into `.wav` files (saved) [audio_wav](https://github.com/ucsd-dsc-arts/dsc160-final-group9/tree/master/data/audio_wav) and eventually into `MIDI` (saved) [audio_mid](https://github.com/ucsd-dsc-arts/dsc160-final-group9/tree/master/data/audio_mid).<br>
- Each of these scraped song's lyrics are saved as individual `rtf` files in [lyrics](https://github.com/ucsd-dsc-arts/dsc160-final-group9/tree/master/data/lyrics).

## Code

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for:

- [audio-to-midi](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/audio-to-midi.ipynb): `.ipynb` file for audio data acquisition and preprocessing from miscellaneous video formats to `.wav` audio format, and eventually conversions into `.mid` MIDI file types
- [MelodyRNN](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/MelodyRNN.ipynb): complete `.ipynb` file with MelodyRNN's modelling and generative tasks
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

## Results

(30 points)

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below:

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

### Audio Generation
<b>MelodyRNN</b>
- In an attempt to compare our generated song from Classical-Piano-Composer with actual songs currently on the <i>TikTok Trendy Music Chart</i>, we utilized a typical and popular song that has been long trending on Tiktok: <i>Meow Meow Meow</i>. This song's main Chorus was extracted and fed into our pre-trained MelodyRNN model to study the song's underlying stylistic components, and generate continuing notes for later use of inspecting if that generated by Classical-Piano-Composer resembles fashions of actual songs on the current chart.
- In order to ensure comparing component is only on musical styles, we made sure that the generated melodic sequence are of the same length. `temperature`, as an important hyper-parameter in MelodyRNN to manipulate how random the output sequence is, is also maintained to be `1.7` throughout all model fittings as well. 
- This first graph is a visualization of the prediction data which according to the model best approximates the notes that when played on the piano at the predicted time frames and with the predicted velocities would produce <i>Meow Meow Meow</i>. This graph is done using <i>Bokeh</i>, and therefore is interactive if you view it in [MelodyRNN.ipynb](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/MelodyRNN.ipynb).<br>
![](results/melody_rnn/bokeh_plot_rendered_meow.png)
- This following second graph plots the generated new <i>Meow Meow Meow</i> (audio file linked [here](results/melody_rnn/rnn_meow.wav)). With time frames all set the same, it can be seen from this comparison, that the generated notes seem to on average each last shorter in time. As we can see that those blocks in this graph tends to be smaller (horizontally) compared to those above. One thing to also notice is how this generated audio shows higher notes to the end of our output sequence than the original rendered song. This might be due to the model learning the overall notes in the training sequence being highly upbeat, and therefore producing an output trend at the end of increasing pitch. However, this might just be a generalized discussion, more precise analysis would be needed for more accurate justifications.<br>
![](results/melody_rnn/bokeh_plot_rnn_meow.png)

## Discussion

(30 points, three to five paragraphs)

The first paragraph should be a short summary describing your results.

The subsequent paragraphs could address questions including:
- Why is this culturally innovative?
- How does your generative computational approach differ from traditional art/music/cultural production?
- How do your results relate to broader social, cultural, economic political, etc., issues?
- What are the ethical concerns for this form of generative art?
- In what future directions could you expand this work?

## Team Roles

- Jou-Ying Lee
	- Wrote Abstract
	- Obtained and processed audio and text data
	- Completed MelodyRNN modelling (MelodyRNN.ipynb)
	- Drafted <i>Results</i> for MelodyRNN in the corresponding section.
	- Finished composition of <i>Technical Notes and Dependencies</i> for <i>audio-to-midi.ipynb</i> and <i>MelodyRNN.ipynb</i>
- Yunlin Tang, yut085@ucsd.edu
- Yupei Zhou, yuz522@ucsd.edu
- Sizhu Chen, sic100@ucsd.edu
- Yuanbo Shi, yus263@ucsd.edu

## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work.
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

Our codes are solely based on Python programming language. The following packages and libraries are therefore all of Python dependencies. While most ipynb files include necessary pip installs within themselves, there are some additional installs needed.<br>
#### [audio-to-midi.ipynb](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/audio-to-midi.ipynb)
`youtube-dl` package enables downloading videos from youtube.com or other video platforms
pip installation can be done by:<br>
<pre><code>sudo -H pip install --upgrade youtube-dl</code></pre>
macOS users can also install with Homebrew:
<pre><code>brew install youtube-dl</code></pre>
#### [MelodyRNN.ipynb](https://github.com/ucsd-dsc-arts/dsc160-final-group9/blob/master/code/MelodyRNN.ipynb)
Processing included in this notebook relies greatly on `Magenta`, especially `magenta.music` for music generation. This library provides numerous Machine Learning Models built with <i>TensorFlow</i>, and therefore they run faster on a GPU. For this reason, this notebook is implemented with commands to be run on <b><i>Google Colab</i></b>.<br>
The following is included in the notebook already, and shall be run for proper execution of later codes.<br>
Setup environment as:
<pre><code>%tensorflow_version 1.x
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob

#!rm -r /content/onsets-frames
!mkdir /content/onsets-frames
!gsutil -q -m cp -R gs://magentadata/models/onsets_frames_transcription/* /content/onsets-frames/
!unzip -o /content/onsets-frames/maestro_checkpoint.zip -d /content/onsets-frames
CHECKPOINT_DIR = '/content/onsets-frames/train'</code></pre>
Important dependendies are to be installed and imported:<br>
`libfluidsynth1` is a real-time MIDI software synthesizer.<br>
`pyfluidsynth` is a Python bindings for FluidSynth, a MIDI synthesizer that uses SoundFont instruments.<br>
`pretty_midi` contains function/classes for handling MIDI data.<br>
`pafy` is a Python library to download YouTube content and retrieve metadata.<br>
`spleeter` is a library and <i>Tensorflow</i> model that achives state-of-the-art source separation from audio files.<br>
`magenta` is a <i>Tensorflow</i> library that explores machine learning in processes of creating art and music.
<pre><code>!apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev ffmpeg  
!pip install pyfluidsynth pretty_midi youtube-dl Pafy
!pip install spleeter

import pafy

if glob.glob('/content/onsets-frames/magenta*.whl'):
  !pip install -q /content/onsets-frames/magenta*.whl
else:
  !pip install -qU magenta</code></pre>

This is the hack to allow python to pick up the newly-installed fluidsynth lib, and is only needed for the hosted Colab environment.
<pre><code>import ctypes.util
orig_ctypes_util_find_library = ctypes.util.find_library
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return orig_ctypes_util_find_library(lib)
ctypes.util.find_library = proxy_find_library</code></pre>
`midi2audio` makes it easy to use MIDI to audio or playback via FluidSynth.<pre><code>!pip install midi2audio</code></pre>
`fluidsynth` is a software synthesizer based on the SoundFont 2 specifications. This additional installation is required for `midi2audio` to properly work for our purpose.<pre><code>!sudo apt-get install fluidsynth</code></pre>

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
	- https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn
	- https://github.com/ytdl-org/youtube-dl
	- https://github.com/tensorflow/magenta
- Blog posts
	- https://tech.uqido.com/2020/02/13/play-it-again-ai-a-look-at-google-magenta-and-machine-learning-for-audio/
- Others
	- https://craffel.github.io/pretty-midi/
	- https://pypi.org/project/pyFluidSynth/
	- https://packages.debian.org/sid/libfluidsynth1
	- https://pythonhosted.org/pafy/
	- https://pypi.org/project/midi2audio/
	- http://www.fluidsynth.org/api/
