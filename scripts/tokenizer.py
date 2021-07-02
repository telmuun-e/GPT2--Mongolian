import sentencepience as spm

spm.SentencePieceTrainer.train(input='../data/cleaned_data/clean_data.txt', model_prefix='mn', vocab_size=50257)