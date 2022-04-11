# -*- coding:utf-8 -*
import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, spans, types):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.spans = spans
        self.types = types

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids, label_mask):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.full_label_ids = full_label_ids
        self.span_label_ids = span_label_ids
        self.type_label_ids = type_label_ids
        self.label_mask = label_mask

def read_examples_from_file(args, data_dir, mode):
    file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            words = item["str_words"]
            labels_ner = item["tags_ner"]
            labels_esi = item["tags_esi"]
            labels_net = item["tags_net"]
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels_ner, spans=labels_esi, types=labels_net))
            guid_index += 1
    examples_src = []
    examples_inter = []
    if mode == "train":
        file_path = os.path.join(data_dir, "{}_{}.json".format(args.src_dataset, mode))
        guid_index = 1
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item["str_words"]
                labels_ner = item["tags_ner"]
                labels_esi = item["tags_esi"]
                labels_net = item["tags_net"]
                examples_src.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels_ner, spans=labels_esi, types=labels_net))
                guid_index += 1

    return examples, examples_src

def convert_examples_to_features(
    tag_to_id,
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = -1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    span_non_id = tag_to_id["span"]["O"]
    type_non_id = tag_to_id["type"]["O"]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        span_label_ids = []
        type_label_ids = []
        label_mask = []
        # print(len(example.words), len(example.labels))
        for word, span_label, type_label in zip(example.words, example.spans, example.types):
            # print(word, label)
            span_label = tag_to_id["span"][span_label]
            type_label = tag_to_id["type"][type_label]
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if len(word_tokens) > 0:
                span_label_ids.extend([span_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                type_label_ids.extend([type_label] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_mask.extend([1] + [0]*(len(word_tokens) - 1))
            # full_label_ids.extend([label] * len(word_tokens))

        # print(len(tokens), len(label_ids), len(full_label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            span_label_ids = span_label_ids[: (max_seq_length - special_tokens_count)]
            type_label_ids = type_label_ids[: (max_seq_length - special_tokens_count)]
            label_mask = label_mask[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        span_label_ids += [pad_token_label_id]
        type_label_ids += [pad_token_label_id]
        label_mask += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            label_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            span_label_ids += [pad_token_label_id]
            type_label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            label_mask += [0]
        else:
            tokens = [cls_token] + tokens
            span_label_ids = [pad_token_label_id] + span_label_ids
            type_label_ids = [pad_token_label_id] + type_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            label_mask += [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            span_label_ids = ([pad_token_label_id] * padding_length) + span_label_ids
            type_label_ids = ([pad_token_label_id] * padding_length) + type_label_ids
            label_mask = ([0] * padding_length) + label_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            span_label_ids += [pad_token_label_id] * padding_length
            type_label_ids += [pad_token_label_id] * padding_length
            label_mask += [0] * padding_length
        
        # print(len(input_ids))
        # print(len(label_ids))
        # print(max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_label_ids) == max_seq_length
        assert len(type_label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("span_label_ids: %s", " ".join([str(x) for x in span_label_ids]))
            logger.info("type_label_ids: %s", " ".join([str(x) for x in type_label_ids]))
            logger.info("label_mask: %s", " ".join([str(x) for x in label_mask]))
        # input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
                label_ids=None, full_label_ids=None, span_label_ids=span_label_ids, type_label_ids=type_label_ids, label_mask=label_mask)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    
    return features

def load_and_cache_examples(args, tokenizer, pad_token_label_id, mode):

    tags_to_id = tag_to_id(args.data_dir, args.dataset)
    tags_to_id_src = tag_to_id(args.data_dir, args.src_dataset)
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "{}_{}.pt".format(
            args.dataset, mode
        ),
    )

    cached_features_file_src = None

    if mode == "train":
        cached_features_file_src = os.path.join(
            args.data_dir,
            "{}_{}.pt".format(
                args.src_dataset, mode
            ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if mode == "train":
            logger.info("Loading source domain features from cached file %s", cached_features_file_src)
            features_src = torch.load(cached_features_file_src)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples, examples_src = read_examples_from_file(args, args.data_dir, mode)
        features = convert_examples_to_features(
            tags_to_id,
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
            sep_token = tokenizer.sep_token,
            sep_token_extra = bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left = bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id = pad_token_label_id,
        )

        features_src = convert_examples_to_features(
            tags_to_id_src,
            examples_src,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
            sep_token = tokenizer.sep_token,
            sep_token_extra = bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left = bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id = pad_token_label_id,
        )
        
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            if mode == "train":
                logger.info("Saving features into cached file %s", cached_features_file_src)
                torch.save(features_src, cached_features_file_src)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_span_label_ids = torch.tensor([f.span_label_ids for f in features], dtype=torch.long)
    all_type_label_ids = torch.tensor([f.type_label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_span_label_ids, all_type_label_ids, all_label_mask, all_ids)
    
    dataset_src = None
    if mode == "train":
        # Convert to Tensors and build dataset
        all_input_ids_src = torch.tensor([f.input_ids for f in features_src], dtype=torch.long)
        all_input_mask_src = torch.tensor([f.input_mask for f in features_src], dtype=torch.long)
        all_segment_ids_src = torch.tensor([f.segment_ids for f in features_src], dtype=torch.long)
        all_span_label_ids_src = torch.tensor([f.span_label_ids for f in features_src], dtype=torch.long)
        all_type_label_ids_src = torch.tensor([f.type_label_ids for f in features_src], dtype=torch.long)
        all_label_mask_src = torch.tensor([f.label_mask for f in features_src], dtype=torch.long)
        all_ids_src = torch.tensor([f for f in range(len(features_src))], dtype=torch.long)

        dataset_src = TensorDataset(all_input_ids_src, all_input_mask_src, all_span_label_ids_src, all_type_label_ids_src, all_label_mask_src, all_ids_src)
    
    return dataset_src, dataset

def get_labels(path=None, dataset_src=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        labels_ner = {}
        labels_span = {}
        labels_type = {}
        non_entity_id = None
        with open(path+dataset+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            spans = data["span"]
            for l, idx in spans.items():
                labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type[idx] = l

        labels_type_src = {}
        with open(path+dataset_src+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            # spans = data["span"]
            # for l, idx in spans.items():
            #     labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type_src[idx] = l

        # if "O" not in labels:
        #     labels = ["O"] + labels
        return labels_span, labels_type, labels_type_src
    else:
        return None, None, None

def tag_to_id(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data # {"ner":{}, "span":{}, "type":{}}
    else:
        return None

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq_type, seq_bio, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    assert len(seq_bio) == len(seq_type)
    spans = tags["span"]
    default = spans["O"]
    bgn = spans["B"]
    inner = spans["I"]
    idx_to_tag = {idx: tag for tag, idx in spans.items()}
    types = tags["type"]
    idx_to_type = {idx: t for t, idx in types.items()}
    chunks = []
    chunks_bio = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq_bio):
        if tok == default and chunk_start is not None:
            chunk = (chunk_start, i)
            chunks_bio.append(chunk)
            if chunk_type != "O":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
            chunk_start = None

        elif tok == bgn:
            if chunk_start is not None:
                chunk = (chunk_start, i)
                chunks_bio.append(chunk)
                if chunk_type != "O":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                chunk_start = None
            chunk_start = i
        else:
            pass
        chunk_type = idx_to_type[seq_type[i].item()]

    if chunk_start is not None:
        chunk = (chunk_start, len(seq_bio))
        chunks_bio.append(chunk)
        if chunk_type != "O":
            chunk = (chunk_type, chunk_start, len(seq_bio))
            chunks.append(chunk)
    return chunks, chunks_bio


if __name__ == '__main__':
    save(args)