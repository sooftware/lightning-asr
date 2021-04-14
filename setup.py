

from distutils.core import setup

setup(
    name='lasr',
    version='latest',
    description='PyTorch Lightning implementaion of Automatic Speech Recognition',
    author='Soohwan Kim',
    author_email='kaki.ai@tunib.ai',
    url='https://github.com/sooftware/lasr',
    install_requires=[
        'torch>=1.4.0',
        'python-Levenshtein',
        'librosa >= 0.7.0',
        'torchaudio',
        'numpy',
        'pandas',
        'astropy',
        'sentencepiece',
        'pytorch-lightning',
    ],
    keywords=['asr', 'speech_recognition', 'pytorch-lightning'],
    python_requires='>=3.7',
)