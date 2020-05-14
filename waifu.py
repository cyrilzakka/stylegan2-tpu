import sys
import os
from pprint import pprint

sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pymen/bin')]
#print(os.path.expanduser("~"))
#sys.path += [os.path.expanduser('~'), 'pymen', 'bin']

import lumen

if 'COLAB_TPU_ADDR' in os.environ:
  os.environ['TPU_NAME'] = 'grpc://' + os.environ['COLAB_TPU_ADDR']

os.environ['NOISY'] = '1'
# --- set resolution and label size here:
label_size = int(os.environ['LABEL_SIZE'])
resolution = int(os.environ['RESOLUTION'])
num_channels = int(os.environ['NUM_CHANNELS'])
model_dir = os.environ['MODEL_DIR']
count = int(os.environ['COUNT']) if 'COUNT' in os.environ else 1
discord_token = os.environ['DISCORD_TOKEN']
grid_image_size = int(os.environ['GRID_SIZE']) if 'GRID_SIZE' in os.environ else 9
batch_size = 1
# ------------------------


import numpy as np
from gensim.models.doc2vec import Doc2Vec
import warnings
warnings.filterwarnings('ignore')

#import gdown
#gdown.download('https://drive.google.com/uc?id=1W0irCj3Ri3APqwHbAr159EoS4k8vdyV9', 'network-apcs128.pkl', quiet=False)
#gdown.download('https://drive.google.com/uc?id=1fvsL3vAFh6FH99zlo7bJxJvz_MMnsTgc', 'danbooru_subset_tags_128.d2v', quiet=False)
#gdown.download('https://drive.google.com/uc?id=1D5rE9WScxjgFQS-T4DoL77NVDkaFty_i', 'danbooru_subset_tags_128.d2v.docvecs.vectors_docs.npy', quiet=False)
model = Doc2Vec.load("danbooru_subset_tags_128.d2v")
import tqdm
from pprint import pprint as pp
from training.networks_stylegan2 import *
from training import misc

import dnnlib
from dnnlib import EasyDict

import tensorflow as tf
import tflex
import os
import numpy as np

tflex.self = globals()

#synthesis_func          = 'G_synthesis_stylegan2'
#kwargs = {'resolution': 512}
#synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)

#sess.reset(os.environ['TPU_NAME']) # don't do this, this breaks the session

train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
G_args    = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
D_args    = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
sched     = EasyDict()                                                     # Options for TrainingSchedule.
grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
tf_config = {'rnd.np_random_seed': 1000}    
label_dtype = np.int64
sched.minibatch_gpu = 1

"""
import pretrained_networks
with tflex.device('/gpu:0'):
    tflex.G, tflex.D, tflex.Gs = pretrained_networks.load_networks('network-apcs128.pkl')
    tflex.G.print_layers()
    tflex.D.print_layers()
"""

def with_session(sess, f, *args, **kws):
  with sess.as_default():
    return f(*args, **kws)

async def with_session_async(sess, f, *args, **kws):
  with sess.as_default():
    return await f(*args, **kws)

def init(session=None, num_channels=None, resolution=None, label_size=None):
  label_size = int(os.environ['LABEL_SIZE']) if label_size is None else label_size
  resolution = int(os.environ['RESOLUTION']) if resolution is None else resolution
  num_channels = int(os.environ['NUM_CHANNELS']) if num_channels is None else num_channels
  dnnlib.tflib.init_tf()

  session = tflex.get_session(session)
  pprint(session.list_devices())

  tflex.set_override_cores(tflex.get_cores())
  with tflex.device('/gpu:0'):
    tflex.G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, **G_args)
    tflex.G.print_layers()
    tflex.Gs, tflex.Gs_finalize = tflex.G.clone2('Gs')
    tflex.Gs_finalize()
    tflex.D = tflib.Network('D', num_channels=num_channels, resolution=resolution, label_size=label_size, **D_args)
    tflex.D.print_layers()
  tflib.run(tf.global_variables_initializer())
  return session

def load_checkpoint(path, session=None, var_list=None):
  if var_list is None:
    var_list = tflex.Gs.trainables
  ckpt = tf.train.latest_checkpoint(path) or path
  assert ckpt is not None
  print('Loading checkpoint ' + ckpt)
  saver = tf.train.Saver(var_list=var_list)
  saver.restore(tflex.get_session(session), ckpt)
  return ckpt

#tflex.state.noisy = False

init()

load_checkpoint(model_dir)

import PIL.Image
import numpy as np
import requests
from io import BytesIO
vocab = set(model.wv.vocab.keys())

def get_waifu(tags, seed = 0, mu = 0, sigma = 0, truncation=None):
    print("Making waifu with ", tags)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation is not None:
        Gs_kwargs.truncation_psi = truncation
    rnd = np.random.RandomState(seed)
    #tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

    tag_set = tags.split()
    test_words = []
    for tag in tag_set:
        if ':' not in tag:
            for i in range(5):
                if str(i) + ':' + tag in vocab:
                    tag = str(i) + ':' + tag
                    break
        if tag != '' and tag in vocab:
            test_words.append(tag)
    labels = model.infer_vector(test_words, alpha=0.025, min_alpha=0.00025, epochs=2000)

    all_seeds = [seed] * batch_size
    all_z = np.stack([np.random.RandomState(seed).randn(*tflex.Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    print(all_z.shape)

    drange_net = [-1, 1]
    if mu != 0 and sigma != 0:
      s = np.random.normal(mu, sigma, 128)
      labels = labels + s
    with tflex.device('/gpu:0'):
      l = np.matrix(labels)
      print(l.shape)
      result = tflex.Gs.run(all_z, l, is_validation=True, randomize_noise=False, minibatch_size=sched.minibatch_gpu)
      if result.shape[1] > 3:
        final = result[:, 3, :, :]
      else:
        final = None
      result = result[:, 0:3, :, :]
      img = misc.convert_to_pil_image(misc.create_image_grid(result, (1, 1)), drange_net)
      #new_im = misc.convert_to_pil_image(result, drange_net)
      img.save('waifu.png')
      return result, img

    #display(new_im)

import discord
import asyncio

client = discord.Client()
token=discord_token
channel_name = 'bot-screenshots'

async def send_picture(channel, image, kind='png', name='test', text=None):
    img = misc.convert_to_pil_image(image, [-1, 1])
    f = BytesIO()
    img.save(f, kind)
    f.seek(0)
    picture = discord.File(f)
    picture.filename = name + '.' + kind
    await channel.send(content=text, file=picture)

@client.event
async def on_ready():
    print('Logged on as {0}!'.format(client.user))
    channel = [x for x in list(client.get_all_channels()) if channel_name in x.name]
    assert len(channel) == 1
    channel = channel[0]
    print(channel)
    #await send_picture(channel, image, kind=kind, name=name, text=text)

import subprocess
def shell(x):
    result = subprocess.run(x, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=isinstance(x, str))
    if len(result.stderr) > 0 and len(result.stdout.strip()) <= 0:
        return result.stderr.decode('utf8')
    else:
        return result.stdout.decode('utf8')

import shlex

def args(msg):
    return shlex.split(msg)[1:]

admins = ['arfa#1551', 'shawwn#3694']

from io import StringIO

def pretty(x):
    if isinstance(x, bytes):
        x = x.decode('utf8')
    if not isinstance(x, str):
        o = StringIO()
        pprint(x, stream=o)
        x = o.getvalue()
    return x

async def respond(to, result):
    if hasattr(to, 'channel'):
        to = to.channel
    lines = pretty(result).splitlines()
    s = []
    n = 0
    trail = None
    async def flush():
        nonlocal n, s
        if n == 0: return
        await to.send(content="```{}```".format('\n'.join(s)))
        n = 0
        s = []
    flushed = 0
    for line in lines:
        if n > 2000 - 6:
            if flushed == 4:
                await to.send(content="```...````")
            elif flushed < 4:
                await flush()
            else:
                trail = s[:]
                n = 0
                s = []
        n += len(line) + 1
        s.append(line)
    await flush()
    if trail is not None:
        s = trail
        await flush()
    return result

@client.event
async def on_message(message):
    tflex.message = message
    print('Message from {0.author}: {0.content}'.format(message))
    try:
        if message.content.startswith("!terminate"):
            await message.channel.send(content="I cannot self-terminate :Kappa:")
        elif message.content.startswith("!shell"):
            if str(message.author) not in admins:
                await message.channel.send(content="no :idk:")
            else:
                cmd = message.content.lstrip('!shell').lstrip()
                await message.channel.send(content="```{}$ {}```".format(os.getcwd(), cmd))
                await message.channel.send(content="```{}```".format(shell(cmd)))
        elif message.content.startswith("!debug"):
            await message.channel.send(content="Breaking in the debugger.. (will hang)")
            import pdb; pdb.set_trace()
        elif message.content.startswith("!quit"):
            await message.channel.send(content="Guess I'll go die then")
            import posix
            posix._exit(0)
        elif message.content.startswith("!ping"):
            await message.channel.send(content="Pong")
        elif message.content.startswith("!reload"):
            argv = args(message.content)
            ckpt = argv[0] if len(argv) > 0 else model_dir
            ckpt = load_checkpoint(model_dir)
            await message.channel.send(content="Loaded checkpoint {}".format(ckpt))
        elif message.content.startswith("!waifu"):
            tags = message.content[6:]
            print("Tags:", tags)
            seed = np.random.randint(10000)
            result, img = get_waifu(tags, seed=seed)
            text = 'Waifu from `{}` with tags `{}` and seed `{}`'.format(model_dir, tags, seed)
            await send_picture(message.channel, misc.create_image_grid(result, (1,1)), kind="png", name='waifu', text=text)
        elif message.content.startswith("!error"):
            argv = args(message.content)
            raise Exception(argv[0] if len(argv) > 0 else 'Heck')
        elif message.content.startswith("! ") or message.channel.name == 'bot-repl' and str(message.author) in admins:
            if str(message.author) not in admins:
                await message.channel.send(content="no :idk:")
            else:
                cmd = message.content.lstrip('! ').lstrip()
                expr = lumen.reader.read_all(lumen.reader.stream(cmd))
                expr = expr[0] if len(expr) <= 0 else (['do'] + expr)
                result = lumen.L_eval(expr)
                #await respond(message, "> " + lumen.L_str(expr))
                await respond(message, result)
    except:
        import traceback
        lines = traceback.format_exc().splitlines()
        msg = """```{}```""".format('\n'.join(lines if len(lines) < 10 else (lines[0:10] + ['...'] + lines[-10:])))
        await message.channel.send(content=msg)

client.run(token)
