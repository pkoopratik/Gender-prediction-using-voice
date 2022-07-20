# predict gender using audio file(.wav file)
made by pratik kalaskar (CSE IIIT jabalpur)
this project is divided into two parts
1)developing model(creating .sav file)
2)real time implementation by using django



# 1)developing model(creating .sav file)
audio source (10.4gb) = http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/
converted this audio source into featues which is names as 'features.csv'
used MFCC to convert this audio singals into featues



conclusion :final model=SVM
SVM classifier model saved with 97% test accuracy to 'finalised_model.sav' using <a href="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/">this dataset</a>.
This has been created for distributing and testing on different languages,accents and on different people to understand the flaws of the model.

### Packages required - (pip install)
 1. librosa
 2. matplotlib
 3. pyaudio
 4. sklearn
 5. pandas
 6. scipy
 7. numpy
 8. wave
 9. pitch

### Steps
 1. Download the full <b>"ML_final"</b> folder only at a specific path (preferred C:\ )
 2. Open 'live_testing.py' and set the path to your downloaded folder (path="C:\ML_final//") 
 3. Run the 'live_testing.py' file and test your results.
