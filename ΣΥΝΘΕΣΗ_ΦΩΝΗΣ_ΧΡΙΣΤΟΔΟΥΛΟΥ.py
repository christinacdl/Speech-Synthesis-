# -*- coding: utf-8 -*-
"""
@author: Christina Christodoulou, LT1200027

"""

# Import necessary libraries
from conch.analysis.formants import FormantTrackFunction
from conch.analysis.pitch import PitchTrackFunction
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''
# Packages needed to be installed:
pip install conch_sounds
pip install librosa
pip install numpy
python -m pip install -U matplotlib
pip install -U scikit-learn

Speech samples were obtained from a two-person interview, a woman and a man.
3 vowels were extracted (/e/,/i/ and /o/), 2 parts from each speaker.
There are 3 folders named "e", "i", "o", which contain 4 wav files, 2 from the female and 2 from the male speaker.
The wav files are named as "e/i/o_0/1/2/3.wav".
The vowels are included in words like "σπίτι" and "σχολές" pronounced by both speakers during the interview.
The rest of the vowels are included in words like "ήδη", "έδειξες", "σχέδια".
'''

# ============================================== QUESTION A ==============================================

# NECESSARY FUNCTIONS

# A function for extracting the desired features and for storing them in proper structures for further processing
def get_pitch_and_formants(f):
    func = PitchTrackFunction(time_step=0.01, min_pitch=75, max_pitch=600)
    pitch = func(f)
    func = FormantTrackFunction(time_step=0.01,
                                window_length=0.025, num_formants=5, max_frequency=5500)
    formants = func(f)
    # sync
    kp = np.array( list( pitch.keys() ) )
    kf = np.array( list( formants.keys() ) )
    
    synced = {}
    for t in kp:
        # find nearest value in formants
        idx = (np.abs( kf - t )).argmin()
        formants_freqs_list = []
        formants_ratios_list = []
        formants_amps_list = []
        current_pitch = pitch[ t ]
        if current_pitch[0] != 0:
            for fmnt in formants[ kf[ idx ] ]:
                if current_pitch[0] is not None and fmnt[0] is not None:
                    formants_freqs_list.append( fmnt[0] )
                    formants_amps_list.append( fmnt[1] )
                    formants_ratios_list.append( fmnt[0]/current_pitch[0] )
            synced[ t ] = {
                'pitch': pitch[ t ],
                'time': t,
                'formants_freqs': formants_freqs_list,
                'formants_amps': formants_amps_list,
                'formants_ratios': formants_ratios_list,
            }
    
    return synced, pitch, formants, kp, kf



# A function for ploting structures
def compount_pitch_formants_plot(file_name, pitch_formants, plt_alias, idx):
    y, sr = librosa.load(file_name, sr=44100)
    n_fft = 8096  
    hop_size = 256
    p = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
    d = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    librosa.display.specshow(d, cmap='gray_r', sr=sr, hop_length=hop_size, x_axis='time', y_axis='linear', ax=plt_alias)
    # and also, if we restrict to speech-relevant frequencies
    lowest_freq = 30
    highest_freq = 5000
    # plt_alias.ylim([lowest_freq, highest_freq])
    plt_alias.set_ylim([lowest_freq, highest_freq])

    # append pitch
    p = []
    t = []
    fmnts = []
    for k in list( pitch_formants.keys() ):
        a = pitch_formants[k]
        p.append( a['pitch'] )
        t.append( a['time'] )
        fmnts.append( a['formants_freqs'] )

    p = np.array( p )
    t = np.array( t )
    plt_alias.plot( t , p , 'r.' )
    for i in range( t.size ):
        for j in range( len(fmnts[i]) ):
            plt_alias.plot( t[i] , fmnts[i][j] , 'b.' )



# A function for forming a matrix that incorporates 5-tuples of formants for each frame and for every recording of a vowel
def isolate_5_formant_stats(s):
    fmnts = np.zeros( ( len(s) , 5 ) )
    nnz = 0
    for i , k in enumerate( list( s.keys() ) ):
        a = s[k]
        tmp_arr = np.array(a['formants_freqs'])
        if tmp_arr.size == 5 and 0 not in tmp_arr:
            fmnts[i, :] = tmp_arr
            nnz += 1
    ret_fmnts = np.zeros( ( nnz , 5 ) )
    nnz = 0
    for i in range( fmnts.shape[0] ):
        if np.sum( fmnts[i,:] ) > 0:
            ret_fmnts[nnz, :] = fmnts[i,:]
            nnz += 1
    return ret_fmnts

#%% VOWELS FOR BOTH SPEAKERS

# Get the pitch and the formants and gather all for both speakers in a list for each vowel separately.

# A list for the /e/ vowel
e_vowel_pitch_formants = []

# A list for the /i/ vowel
i_vowel_pitch_formants = []

# A list for the /o/ vowel
o_vowel_pitch_formants = []

for i in range(4):
    e, _, _, _, _ = get_pitch_and_formants( 'audio files/e/e_' + str(i) + '.wav' )
    e_vowel_pitch_formants.append( e )
#print(e_vowel_pitch_formants)
    ai, _, _, _, _ = get_pitch_and_formants( 'audio files/i/i_' + str(i) + '.wav' )
    i_vowel_pitch_formants.append( ai )
#print(i_vowel_pitch_formants)
    o, _, _, _, _ = get_pitch_and_formants( 'audio files/o/o_' + str(i) + '.wav' )
    o_vowel_pitch_formants.append( o )
#print(o_vowel_pitch_formants)
          
#%% Plots for the /e/ vowel for both speakers 

fig,a =  plt.subplots(2,2)
fig.suptitle('/e/ vowel for both speakers')

for i in range( 4 ):
    file_name = 'audio files/e/e_' + str(i) + '.wav' 
    pitch_formants = e_vowel_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i//2][i%2], i)
    
fig.savefig('figures/e_vowel.png', dpi=500)

#%% Plots for the /i/ vowel for both speakers 

fig,a =  plt.subplots(2,2)
fig.suptitle('/i/ vowel for both speakers')

for i in range( 4 ):
    file_name = 'audio files/i/i_' + str(i) + '.wav'
    pitch_formants = i_vowel_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i//2][i%2], i)

fig.savefig('figures/i_vowel.png', dpi=500)

#%% Plots for the /o/ vowel for both speakers 

fig,a =  plt.subplots(2,2)
fig.suptitle('/o/ vowel for both speakers')

for i in range( 4 ):
    file_name = 'audio files/o/o_' + str(i) + '.wav'
    pitch_formants = o_vowel_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i//2][i%2], i)

fig.savefig('figures/o_vowel.png', dpi=500)

#%% Boxplot for the /e/ vowel for both speakers

e_stacked = isolate_5_formant_stats( e_vowel_pitch_formants[0] )
for i in range( 1, 4, 1 ):
    e_stacked = np.vstack( ( e_stacked , isolate_5_formant_stats( e_vowel_pitch_formants[i] ) ) )

plt.boxplot( e_stacked )
plt.title('/e/ vowel for both speakers')
plt.savefig('figures/e_vowel_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# The F4 has wider distribution than the rest of the formants. 
# They are a few outliers located outside the whiskers of the box plots of F3 and F5. 
# F5 seems symmetric, F2, F3 and F4 are negatively skewed, while F1 is positively skewed.

# %% Boxplot for the /i/ vowel for both speakers

i_stacked = isolate_5_formant_stats( i_vowel_pitch_formants[0] )
for i in range( 1, 4, 1 ):
    i_stacked = np.vstack( ( i_stacked , isolate_5_formant_stats( i_vowel_pitch_formants[i] ) ) )

plt.boxplot( i_stacked )
plt.title('/i/ vowel for both speakers')
plt.savefig('figures/i_vowel_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# There are some outliers located outside the whiskers of the box plots of F1 and F4.
# F1 and F3 are positively skewed, while F2,F4,F5 are negatively skewed.

#%% Boxplot for the /o/ vowel for both speakers

o_stacked = isolate_5_formant_stats( o_vowel_pitch_formants[0] )
for i in range( 1, 4, 1 ):
    o_stacked = np.vstack( ( o_stacked , isolate_5_formant_stats( o_vowel_pitch_formants[i] ) ) )

plt.boxplot( o_stacked )
plt.title('/o/ vowel for both speakers')
plt.savefig('figures/o_vowel_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# There are some outliers located outside the whiskers of the box plots.
# F1 is symmetric, F2 and F3 are positively skewed and F4, F5 are negatively skewed. 

#%% A combined boxplot for all 3 vowels for both speakers

idx = np.min( np.array( [e_stacked.shape[0] , i_stacked.shape[0] , o_stacked.shape[0] ] ) )
combined = np.hstack( ( e_stacked[:idx,:] , i_stacked[:idx,:] , o_stacked[:idx,:] ) )

plt.boxplot( combined )
plt.title('The 3 vowels /e/,/i/,/o/ for both speakers')
plt.savefig('figures/combined_vowels_boxplot.png', dpi=500)

# The combined boxplot of all formants shows that there are significant differences in the formants layout. 
# F1 is distinctive for all vowel recordings, especially for the /i/ and /o/ vowel which has smaller distribution.

#%% A combined plot for all 3 vowels for both speakers

letters = [ 'e' , 'i' , 'o' ]
colors = [ 'red' , 'green' , 'blue' ]
compv = np.vstack( ( e_stacked[:idx,:] , i_stacked[:idx,:] , o_stacked[:idx,:] ) )
idxs = np.hstack( ( 0*np.ones( idx ) , 1*np.ones( idx ) , 2*np.ones( idx ) ) ).astype(int)

fig,a =  plt.subplots()

for i in range( compv.shape[0] ):
    a.plot( compv[i,0] , compv[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (compv[i,0] , compv[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_vowels_speakers.png' , dpi=500 )

# Ploting the first two formants (F1 on x and F2 on y) shows that they are scattered and, thus, it is difficult to separate the vowel recordings.

#%% A combined PCA for all 3 vowels for both speakers

pca = PCA(n_components=2)
pca.fit(compv)
transformed = pca.transform(compv)

fig,a =  plt.subplots()

for i in range( compv.shape[0] ):
    a.plot( transformed[i,0] , transformed[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (transformed[i,0] , transformed[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_vowels_PCA_speakers.png' , dpi=500 )

# Even with Principal Component Analysis it is not easy to separate the vowel recordings and establish for each one a vowel space,since they are very scattered. 

#%%======== VOWELS FOR EACH SPEAKER - FEMALE SPEAKER ========

# Get the pitch and the formants and gather for each speaker separately in a list.
# The first two audio files belong to the female speaker and the rest to the male speaker.

# A list for the /e/ vowel
e_female_pitch_formants = []

# A list for the /i/ vowel
i_female_pitch_formants = []

# A list for the /o/ vowel
o_female_pitch_formants = []

for i in range(2):
    e_fem, _, _, _, _ = get_pitch_and_formants( 'audio files/e/e_' + str(i) + '.wav' )   
    e_female_pitch_formants.append( e_fem )
#print(e_female_pitch_formants)
    ai_fem, _, _, _, _ = get_pitch_and_formants( 'audio files/i/i_' + str(i) + '.wav' )
    i_female_pitch_formants.append( ai_fem )
#print(i_female_pitch_formants)
    o_fem, _, _, _, _ = get_pitch_and_formants( 'audio files/o/o_' + str(i) + '.wav' )
    o_female_pitch_formants.append( o_fem )
#print(o_female_pitch_formants)

#%% Plots for the /e/ vowel for the female speaker

fig,a =  plt.subplots(1,2)
fig.suptitle('/e/ vowel female')

for i in range( 2 ):
    file_name = 'audio files/e/e_' + str(i) + '.wav' 
    pitch_formants = e_female_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)
    
fig.savefig('figures/e_female.png', dpi=500)

# The first sound file of the /e/ vowel is longer in duration than the second.

#%% Plots for the /i/ vowel for the female speaker
 
fig,a =  plt.subplots(1,2)
fig.suptitle('/i/ vowel female')

for i in range( 2 ):
    file_name = 'audio files/i/i_' + str(i) + '.wav'
    pitch_formants = i_female_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)

fig.savefig('figures/i_female.png', dpi=500)

#%% Plots for the /o/ vowel for the female speaker

fig,a =  plt.subplots(1,2)
fig.suptitle('/o/ vowel female')

for i in range( 2 ):
    file_name = 'audio files/o/o_' + str(i) + '.wav'
    pitch_formants = o_female_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)

fig.savefig('figures/o_female.png', dpi=500)

#%% Boxplot for the /e/ vowel for the female speaker

e_stacked_fem = isolate_5_formant_stats( e_female_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    e_stacked_fem = np.vstack( ( e_stacked_fem , isolate_5_formant_stats( e_female_pitch_formants[i] ) ) )

plt.boxplot( e_stacked_fem )
plt.title('/e/ vowel female')
plt.savefig('figures/e_female_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# F4 longer, which means it has wider distibution than the rest. F1 has the smaller distribution.
# F1, F2, F3 and F4 are negatively skewed, while F5 is positively skewed.
# There are a few outliers in F1 and F2.

# %% Boxplot for the /i/ vowel for the female speaker

i_stacked_fem = isolate_5_formant_stats( i_female_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    i_stacked_fem = np.vstack( ( i_stacked_fem , isolate_5_formant_stats( i_female_pitch_formants[i] ) ) )

plt.boxplot( i_stacked_fem )
plt.title('/i/ vowel female')
plt.savefig('figures/i_female_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# There are some outliers located outside the whiskers of the box plots mainly in F1 and F3.
# The F2 seems symmetric, F3 is negatively skewed, F4 and F5 are positively skewed. 
# F1 is distinctive, it has very small distribution with many outliers.

#%% Boxplot for the /o/ vowel for the female speaker

o_stacked_fem = isolate_5_formant_stats( o_female_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    o_stacked_fem = np.vstack( ( o_stacked_fem , isolate_5_formant_stats( o_female_pitch_formants[i] ) ) )

plt.boxplot( o_stacked_fem )
plt.title('/o/ vowel female')
plt.savefig('figures/o_female_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# There are many outliers located outside the whiskers of the box plots.
# The F4 and F5 seem symmetric, F2 and F3 are positively skewed, F1 seems negatively skewed. 
# F1 is distinctive, since it is really small.

#%% A combined boxplot for all 3 vowels for the female speaker

idx = np.min( np.array( [e_stacked_fem.shape[0] , i_stacked_fem.shape[0] , o_stacked_fem.shape[0] ] ) )
combined_fem = np.hstack( ( e_stacked_fem[:idx,:] , i_stacked_fem[:idx,:] , o_stacked_fem[:idx,:] ) )

plt.boxplot( combined_fem )
plt.title('The 3 vowels /e/,/i/,/o/ female')
plt.savefig('figures/combined_female_boxplot.png', dpi=500)

# The combined boxplot of all formants shows that there are significant differences in the formants layout. 
# The F4 in /e/ vowel is distinctive because it is largest in size. 
# The F1 is distinctive in all vowels, since it has the smallest distribution.
# The F3 in /i/ and /o/ vowels has also very small distribution. 

#%% A combined plot for all 3 vowels for the female speaker

letters = [ 'e' , 'i' , 'o' ]
colors = [ 'blue' , 'red' , 'purple' ]
compv1 = np.vstack( ( e_stacked_fem[:idx,:] , i_stacked_fem[:idx,:] , o_stacked_fem[:idx,:] ) )
idxs = np.hstack( ( 0*np.ones( idx ) , 1*np.ones( idx ) , 2*np.ones( idx ) ) ).astype(int)

fig,a =  plt.subplots()

for i in range( compv1.shape[0] ):
    a.plot( compv1[i,0] , compv1[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (compv1[i,0] , compv1[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_female_plot.png' , dpi=500 )

# Ploting the first two formants (F1 on x and F2 on y) shows that the formants are a little scattered, but can be distinguished. 

#%% A combined PCA for all 3 vowels for the female speaker

pca = PCA(n_components=2)
pca.fit(compv1)
transformed1 = pca.transform(compv1)

fig,a =  plt.subplots()

for i in range( compv1.shape[0] ):
    a.plot( transformed1[i,0] , transformed1[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (transformed1[i,0] , transformed1[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_female_PCA.png' , dpi=500 )

#%%======== VOWELS FOR EACH SPEAKER - MALE SPEAKER ========

# Get the pitch and the formants and gather for each speaker separately in a list.
# The first two audio files belong to the female speaker and the rest to the male speaker.

# A list for the /e/ vowel
e_male_pitch_formants = []

# A list for the /i/ vowel
i_male_pitch_formants = []

# A list for the /o/ vowel
o_male_pitch_formants = []

for i in range(2,4):
    e_male, _, _, _, _ = get_pitch_and_formants( 'audio files/e/e_' + str(i) + '.wav' )
    e_male_pitch_formants.append( e_male )
#print(e_male_pitch_formants)
    ai_male, _, _, _, _ = get_pitch_and_formants( 'audio files/i/i_' + str(i) + '.wav' )
    i_male_pitch_formants.append( ai_male )
#print(i_male_pitch_formants)
    o_male, _, _, _, _ = get_pitch_and_formants( 'audio files/o/o_' + str(i) + '.wav' )
    o_male_pitch_formants.append( o_male )
#print(o_male_pitch_formants)

#%% Plots for the /e/ vowel for the male speaker

fig,a =  plt.subplots(1,2)
fig.suptitle('/e/ vowel male')

for i in range( 2 ):
    file_name = 'audio files/e/e_' + str(i) + '.wav' 
    pitch_formants = e_male_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)
    
fig.savefig('figures/e_male.png', dpi=500)

#%% Plots for the /i/ vowel for the male speaker 

fig,a =  plt.subplots(1,2)
fig.suptitle('/i/ vowel male')

for i in range( 2 ):
    file_name = 'audio files/i/i_' + str(i) + '.wav'
    pitch_formants = i_male_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)

fig.savefig('figures/i_male.png', dpi=500)

#%% Plots for the /o/ vowel for the male speaker

fig,a =  plt.subplots(1,2)
fig.suptitle('/o/ vowel male')

for i in range( 2 ):
    file_name = 'audio files/o/o_' + str(i) + '.wav'
    pitch_formants = o_male_pitch_formants[i]
    compount_pitch_formants_plot(file_name, pitch_formants, a[i], i)

fig.savefig('figures/o_male.png', dpi=500)

#%% Boxplot for the /e/ vowel for the male speaker

e_stacked_male = isolate_5_formant_stats( e_male_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    e_stacked_male = np.vstack( ( e_stacked_male , isolate_5_formant_stats( e_male_pitch_formants[i] ) ) )

plt.boxplot( e_stacked_male )
plt.title('/e/ vowel male')
plt.savefig('figures/e_male_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# F1 and F2 have the smallest distribution. 
# There are some outliers located outside the whiskers of the box plots in F1,F3 and F5.
# The F2,F3 and F5 seem symmetric, F1 seems negatively skewed, while the F4 slightly positively skewed. 

# %% Boxplot for the /i/ vowel for the male speaker

i_stacked_male = isolate_5_formant_stats( i_male_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    i_stacked_male = np.vstack( ( i_stacked_male , isolate_5_formant_stats( i_male_pitch_formants[i] ) ) )

plt.boxplot( i_stacked_male )
plt.title('/i/ vowel male')
plt.savefig('figures/i_male_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# F2 is longer, which means it has wider distibution than the rest. F1 and F3 are less dispersed.
# There are some outliers located outside the whiskers of the box plots in F1,F3,F4 and F5.
# The F4 seems symmetric, F1, F2, F3 and F5 are negatively skewed.

#%% Boxplot for the /o/ vowel for the male speaker

o_stacked_male = isolate_5_formant_stats( o_male_pitch_formants[0] )
for i in range( 1, 2, 1 ):
    o_stacked_male = np.vstack( ( o_stacked_male , isolate_5_formant_stats( o_male_pitch_formants[i] ) ) )

plt.boxplot( o_stacked_male )
plt.title('/o/ vowel male')
plt.savefig('figures/o_male_boxplot.png', dpi=500)

# There are differences between the formants. The median lines of the box plots are not equal.
# The distribution of each formant is different as can be seen from the interquartile ranges.
# F2 is longer, which means it has wider distibution than the rest.
# There are a few outliers located outside the whisker of the box plot in F4. 
# The F5 and F1 seem symmetric, F3 and F4 are negatively skewed, F2 seems positively skewed. 

#%% A combined boxplot for all 3 vowels for the male speaker

idx = np.min( np.array( [e_stacked_male.shape[0] , i_stacked_male.shape[0] , o_stacked_male.shape[0] ] ) )
combined_male = np.hstack( ( e_stacked_male[:idx,:] , i_stacked_male[:idx,:] , o_stacked_male[:idx,:] ) )

plt.boxplot( combined_male )
plt.title('The 3 vowels /e/,/i/,/o/ male')
plt.savefig('figures/combined_male_boxplot.png', dpi=500)

# The combined boxplot of all formants shows that there are significant differences in the formants layout. 
# It seems that F1 are distinguished from the other formants for all the vowels.
# The F1 are less dispersed (smaller in size) in all vowels.
# F2 in the /i/ vowel and the F4 in the /e/ vowel are distinguished since they have wider distibution than the rest.

#%% A combined plot for all 3 vowels for the male speaker

letters = [ 'e' , 'i' , 'o' ]
colors = [ 'blue' , 'red' , 'purple' ]
compv2 = np.vstack( ( e_stacked_male[:idx,:] , i_stacked_male[:idx,:] , o_stacked_male[:idx,:] ) )
idxs = np.hstack( ( 0*np.ones( idx ) , 1*np.ones( idx ) , 2*np.ones( idx ) ) ).astype(int)

fig,a =  plt.subplots()

for i in range( compv2.shape[0] ):
    a.plot( compv2[i,0] , compv2[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (compv2[i,0] , compv2[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_male_plot.png' , dpi=500 )

# Ploting the first two formants (F1 on x and F2 on y) shows that F2 is enough for linearly separating the /e/ vowel recordings
# by drawing an horizontical line.
# The formants for the other vowels can be distinguished, 
# but not by drawing an horizontical or vertical line since they are dispersed. 

#%% A combined PCA for all 3 vowels for the male speaker

pca = PCA(n_components=2)
pca.fit(compv2)
transformed2 = pca.transform(compv2)

fig,a =  plt.subplots()

for i in range( compv2.shape[0] ):
    a.plot( transformed2[i,0] , transformed2[i,1] , '.', color=colors[idxs[i]] )
    a.annotate( letters[ idxs[i] ] , (transformed2[i,0] , transformed2[i,1]) , color = colors[ idxs[i] ] )

fig.show()
fig.savefig( 'figures/combined_male_PCA.png' , dpi=500 )

#%%
# ============================================= QUESTION B ============================================

# Import necessary libraries
import numpy as np
import numpy.matlib
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# I ran the code kernel by kernel for each speaker separately.

#%% ====== MALE SPEAKER ======

# Synthesis of the /o/ vowel 

# A demo file, before running for a batch of files
file_name_male = 'audio files/e/e_2.wav'
   
# Load the waveform
y, sr = librosa.load(file_name_male, sr=44100)

# Find the mean pitch
pitch = librosa.yin(y, 50, 1000)
fr = np.mean(pitch)

# Apply fft
n_fft = 8096
hop_size = 1024

p = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
d = librosa.amplitude_to_db( np.abs(p), ref=np.max )
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Get the mean of spectral amplitude across all time instances
m = np.mean( d , axis=1 )
# and place it on 0 level
m -= np.min( m )

# A helper function - moving average for smoothing out the spectrum
def moving_average(x, w):
    x = np.hstack( ( x[0]*np.ones(w//2) , x , x[-1]*np.ones(w//2-1) ) )
    ma = np.convolve(x, np.ones(w), 'valid') / w
    return ma

# apply moving average to spectrum and plot both
a = moving_average(m,50)
a /= np.max( a )

plt.plot( freqs , m/np.max(m) )
plt.plot( freqs , a )
plt.savefig( 'audio files/speech synthesis/figures/fig_male_1.png' , dpi=500 )

#%% Use the moving average curve as spectral filter for formants - simulating the vocal tract component

f = numpy.matlib.repmat( a , p.shape[1], 1 ).T
plt.imshow(f, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_male_2.png' , dpi=500 )

#%% Make buzzing component - simulated with a sawtooth waveform

n = np.arange( y.size )
x = (n%(sr/fr))/(sr/fr)

# Apply fft to it - for applying ifft afterwards
xp = librosa.stft(x, n_fft=n_fft, hop_length=hop_size)

# Apply the formants filter
px = xp*f
x1 = librosa.istft(px, hop_length=hop_size)

dpx = librosa.amplitude_to_db( np.abs(px), ref=np.max )
plt.imshow(dpx, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_male_3.png' , dpi=500 )

#%% Show recording spectrum for comparison

plt.imshow(d, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_male_4.png' , dpi=500 )

#%% Speech synthesis of a vowel for the male speaker

file_names_male = ['audio files/e/e_2.wav','audio files/e/e_3.wav']

sythesized = np.zeros(0)
recorded = np.zeros(0)


for i, file_name in enumerate(file_names_male):
    y, sr = librosa.load(file_name, sr=44100)
    
    pitch = librosa.yin(y, 50, 1000)
    fr = np.mean(pitch)
    
    n_fft = 8096
    hop_size = 256
    
    p = librosa.stft(y, n_fft=n_fft, hop_length=hop_size)
    d = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    m1 = np.mean( d , axis=1 )
    m1 -= np.min( m1 )
    
    def moving_average(x, w):
        x = np.hstack( ( x[0]*np.ones(w//2) , x , x[-1]*np.ones(w//2-1) ) )
        ma = np.convolve(x, np.ones(w), 'valid') / w
        return ma
    
    a_male = moving_average(m1,50)
    a_male /= np.max( a_male )
    
    # filter
    f = numpy.matlib.repmat( a , p.shape[1], 1 ).T
    
    # make saw
    n = np.arange( y.size )
    x = (n%(sr/fr))/(sr/fr)
    xp = librosa.stft(x, n_fft=n_fft, hop_length=hop_size)
    
    px = xp*f
    x1 = librosa.istft(px, hop_length=hop_size)
    
    dpx = librosa.amplitude_to_db( np.abs(px), ref=np.max )
    
    sythesized = np.hstack( (sythesized,x1) )
    recorded = np.hstack( (recorded,y) )

sd.play( recorded, sr )
time.sleep( recorded.size/sr + 0.5 )
sd.play( sythesized , sr )
 
#%%  ====== FEMALE SPEAKER ======

# A demo file, before running for a batch of files
file_name_female = 'audio files/i/i_0.wav'
      
# Load the waveform
y1, sr = librosa.load(file_name_female, sr=44100)

# Find the mean pitch
pitch = librosa.yin(y1, 50, 1000)
fr = np.mean(pitch)

# Apply fft
n_fft = 8096
hop_size = 1024

p1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_size)
d1 = librosa.amplitude_to_db( np.abs(p1), ref=np.max )
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# Get the mean of spectral amplitude across all time instances
m1 = np.mean( d1 , axis=1 )
# and place it on 0 level
m1 -= np.min( m1 )

# A helper function - moving average for smoothing out the spectrum
def moving_average(x, w):
    x = np.hstack( ( x[0]*np.ones(w//2) , x , x[-1]*np.ones(w//2-1) ) )
    ma = np.convolve(x, np.ones(w), 'valid') / w
    return ma

# apply moving average to spectrum and plot both
a = moving_average(m1,50)
a /= np.max( a )

plt.plot( freqs , m1/np.max(m1) )
plt.plot( freqs , a )
plt.savefig( 'audio files/speech synthesis/figures/fig_female_1.png' , dpi=500 )

#%% Use the moving average curve as spectral filter for formants - simulating the vocal tract component

f = numpy.matlib.repmat( a , p1.shape[1], 1 ).T
plt.imshow(f, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_female_2.png' , dpi=500 )

#%% Make buzzing component - simulated with a sawtooth waveform

n = np.arange( y1.size )
x = (n%(sr/fr))/(sr/fr)

# Apply fft to it - for applying ifft afterwards
xp = librosa.stft(x, n_fft=n_fft, hop_length=hop_size)

# Apply the formants filter
px = xp*f
x1 = librosa.istft(px, hop_length=hop_size)

dpx = librosa.amplitude_to_db( np.abs(px), ref=np.max )
plt.imshow(dpx, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_female_3.png' , dpi=500 )

#%% Show recording spectrum for comparison

plt.imshow(d1, origin='lower', aspect='auto')
plt.savefig( 'audio files/speech synthesis/figures/fig_female_4.png' , dpi=500 )

#%% Speech synthesis of a vowel for the female speaker

file_names_female = ['audio files/i/i_0.wav','audio files/i/i_1.wav']

sythesized = np.zeros(0)
recorded = np.zeros(0)


for i, file_name in enumerate(file_names_female):
    y2, sr = librosa.load(file_name, sr=44100)
    
    pitch = librosa.yin(y2, 50, 1000)
    fr = np.mean(pitch)
    
    n_fft = 8096
    hop_size = 256
    
    p = librosa.stft(y2, n_fft=n_fft, hop_length=hop_size)
    d = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    m = np.mean( d , axis=1 )
    m -= np.min( m )
    
    def moving_average(x, w):
        x = np.hstack( ( x[0]*np.ones(w//2) , x , x[-1]*np.ones(w//2-1) ) )
        ma = np.convolve(x, np.ones(w), 'valid') / w
        return ma
    
    a = moving_average(m,50)
    a /= np.max( a )
    
    # filter
    f = numpy.matlib.repmat( a , p.shape[1], 1 ).T
    
    # make saw
    n = np.arange( y2.size )
    x = (n%(sr/fr))/(sr/fr)
    xp = librosa.stft(x, n_fft=n_fft, hop_length=hop_size)
    
    px = xp*f
    x1 = librosa.istft(px, hop_length=hop_size)
    
    dpx = librosa.amplitude_to_db( np.abs(px), ref=np.max )
    
    sythesized = np.hstack( (sythesized,x1) )
    recorded = np.hstack( (recorded,y2) )

# Play and listen to the sound 
sd.play( recorded, sr )
time.sleep( recorded.size/sr + 0.5 )
sd.play( sythesized , sr )