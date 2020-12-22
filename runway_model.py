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
                   read_audio,
                   read_batch,
                   prepare_model_input)

def wav_to_text(f='test.wav'):
  batch = read_batch([f])
  input = prepare_model_input(batch, device=device)
  output = model(input)
  return decoder(output[0].cpu())



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


@runway.setup(options={'checkpoint': runway.file(extension='.model',description='checkpoint file')})
def setup(opts):
    models = OmegaConf.load('models.yml')
    device = torch.device('cuda')   
    model, decoder = init_jit_model(opts['checkpoint'], device=device)
    language = "English" 

    use_VAD = "Yes" 

    return model, decoder


@runway.command('translate', inputs={'source_audio': runway.file(extension='.wav', description='input sound file to be translated'),}, outputs={'text': runway.text(description='output text')})
def translate(model, inputs):
    test_files = glob(inputs['source_audio'])  
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(random.sample(batches, k=1)[0]),
                            device=device)
    output = model(input)
    return decoder(example.cpu())




if __name__ == '__main__':
    runway.run(port=8889)
