# Fine-tuned BERT-base-uncased pre-trained model for Indonesian-English hate comments sentiment analysis

My first project in Natural Language Processing (NLP), where I fine-tuned a bert-base-uncased model to classify hate comments in a bilingual (Indonesian-English) dataset. This project focuses on sentiment analysis to detect toxic, offensive, or hateful language commonly found in social media and online platforms.

## TODO

✅ Uses `bert-base-multilingual-uncased`, a widely used multilingual model.\
✅ Clean Dataset class for handling data.\
✅ Uses Hugging Face's Trainer API — very efficient.\
✅ Includes training and evaluation splits.\
✅ Saves the model and tokenizer.\

## ✅ INSTALL REQUIREMENTS

Install required dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## ✅ ADD BERT virtual env

write the command below

```sh
# ✅ Create and activate a virtual environment
python -m venv bert-env
source bert-env/bin/activate    # On Windows use: bert-env\Scripts\activate
```

## ✅ INSTALL CUDA

Check if your GPU supports CUDA:

```sh
nvidia-smi
```

Then:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

## 🔧 HOW TO USE

- Check your device and CUDA availability:

```sh
python check_device.py
```

> :warning: Using CPU is not advisable, prefer check your CUDA availability.

- Train the model:

```sh
python scripts/train.py
```

> :warning: Remove unneeded checkpoint in models/pretrained to save your storage after training

- Run prediction:

```sh
python scripts/predict.py
```

✅ Dataset Location: `data/dataset.csv`, modify the dataset to enhance the model based on your needs.

## License and Usage

This repository is intended for **research and educational purposes only**.  
**Commercial use is strictly prohibited.**

If you are interested in commercial licensing, please contact developerfauzan@gmail.com.

Creative Commons Attribution NonCommercial (CC-BY-NC)

Copyright 2025 Muhammad Fauzan (fzn0x)

---

Leave a ⭐ if you think this project is helpful, contributions are welcome.

> 🚫 This repository is for **research and educational purposes only**. Commercial use is not allowed.

```
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Dataset (MIX): https://github.com/abusifyid/Indonesian-Multimodal-Hate-Speech-Dataset
