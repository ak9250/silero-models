import runway
import numpy as np
import argparse
import torch
from torchvision import transforms
import os.path
import torch
import random
from glob import glob
from omegaconf import OmegaConf
from utils import (init_jit_model, 
                   split_into_batches,
                   read_batch,
                   prepare_model_input)
from IPython.display import display, Audio
import torch
import random
from glob import glob
from omegaconf import OmegaConf
from utils import (init_jit_model, 
                   split_into_batches,
                   read_audio,
                   read_batch,
                   prepare_model_input)
from colab_utils import (record_audio,
                         audio_bytes_to_np,
                         upload_audio)

def wav_to_text(f='test.wav'):
  batch = read_batch([f])
  input = prepare_model_input(batch, device=device)
  output = model(input)
  return decoder(output[0].cpu())

models = OmegaConf.load('models.yml')

device = torch.device('gpu')   # you can use any pytorch device
model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)

#@markdown { run: "auto" }

language = "English" #@param ["English", "German", "Spanish"]

#@markdown { run: "auto" }

use_VAD = "Yes" #@param ["Yes", "No"]

#@markdown Either record audio from microphone or upload audio from file (.mp3 or .wav) { run: "auto" }


def _apply_vad(audio, boot_time=0, trigger_level=9, **kwargs):
  print('\nVAD applied\n')
  vad_kwargs = dict(locals().copy(), **kwargs)
  vad_kwargs['sample_rate'] = sample_rate
  del vad_kwargs['kwargs'], vad_kwargs['audio']
  audio = vad(torch.flip(audio, ([0])), **vad_kwargs)
  return vad(torch.flip(audio, ([0])), **vad_kwargs)

def _recognize(audio):
  display(Audio(audio, rate=sample_rate, autoplay=True))
  if use_VAD == "Yes":
    audio = _apply_vad(audio)
  wavfile.write('test.wav', sample_rate, (32767*audio).numpy().astype(np.int16))
  transcription = wav_to_text()
  print('\n\nTRANSCRIPTION:\n')
  print(transcription)


@runway.setup(options={'checkpoint': runway.file(extension='.pkl',description='checkpoint file')})
def setup(opts):
    path = Path(".")
    learn=load_learner(path, opts['checkpoint'])
    return learn


@runway.command('translate', inputs={'source_audio': runway.file(description='input image to be translated'),}, outputs={'image': runway.text(description='output text')})
def translate(learn, inputs):
    test_files = glob(inputs['source_audio'])  # replace with your data
    batches = split_into_batches(test_files, batch_size=10)
    # transcribe a set of files
    input = prepare_model_input(read_batch(random.sample(batches, k=1)[0]),
                            device=device)
    output = model(input)
    # for example in output:
    return decoder(example.cpu())




if __name__ == '__main__':
    runway.run(port=8889)
