from model.config import Config
from model.data_utils import GloveDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
    """Procedure to build data

    This procedure iterates over the SemEval dataset and builds a vocabulary 
    of words and tags, then writes them to a file. Each word is labelled by 
    an ID. The GloVe vectors of the words are then extracted and stored
    in a numpy array. The word id is used to index into that numpy array.

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators for the dev, test and training files
    dev   = GloveDataset(config.filename_dev, processing_word)
    test  = GloveDataset(config.filename_test, processing_word)
    train = GloveDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    #find the intersection between the vocabs from the chosen dataset and GloVe
    vocab = vocab_words & vocab_glove
    #adds the unknown and numeric value to the vocab
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # export the trimmed glove vectors in a compressed file.
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # # Build and save char vocab
    # train = GloveDataset(config.filename_train)
    # vocab_chars = get_char_vocab(train)
    # write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
