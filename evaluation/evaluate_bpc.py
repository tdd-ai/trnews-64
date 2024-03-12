import math
from utils import *

log2 = 0.6931471805599453


def main(args):

    model = args.model
    batch_size = args.batch_size

    print(f"Loading {model}...")
    llm = load_model(model)
    vocab_size = get_vocab_size(llm)

    if args.context_size is not None:
        context_size = args.context_size
    else:
        context_size = get_context_size(llm)

    print(f"{type(llm)} with vocab_size={vocab_size}")

    if args.wikitext:
        print(f"Loading wikitext-2.test.raw...")
        articles = load_wikitext()
        print(f"Read {len(articles)} articles from wikitext-2.test.raw")
    elif args.contamination:
        print(f"Loading trnews.cont.raw...")
        articles = load_raw(args.contamination)
        print(f"Read {len(articles)} articles from trnews.cont.raw")
    else:
        print(f"Loading trnews-64.test.raw...")
        articles = load_trnews()
        print(f"Read {len(articles)} articles from trnews-64.test.raw")

    print(f"Tokenizing...")
    batches, tokens, nchars = tokenize(
        model,
        articles,
        batch_size=batch_size,
        bos_id=args.bos_id,
        context_size=context_size,
    )
    ntokens = sum(len(t) for t in tokens) - len(
        tokens
    )  # -len(tokens) for added bos tokens

    print(f"Got {ntokens} tokens from {nchars} chars")
    print(f"Got {len(batches)} batches with batch_size={batch_size}")
    print(f"tokenizer encodes {nchars/ntokens} chars per token.")

    (sum_nll, ntokens_predicted, article_losses) = nll(llm, batches)

    assert (
        int(ntokens_predicted) == ntokens
    ), f"Expected {ntokens} tokens, predicted {ntokens_predicted} tokens."

    sum_bits = sum_nll / log2
    bits_per_char = sum_bits / nchars
    token_ppl = math.exp(sum_nll / ntokens_predicted)

    print(
        f"""Summary:
model\t= {model}
ntokens\t= {ntokens_predicted}
nchars\t= {nchars}
nvocabs\t= {vocab_size}
sum_nll\t= {sum_nll}
tkn_ppl\t= {token_ppl}
bpc\t= {bits_per_char}"""
    )

    if args.save_bpc is not None:
        print(f"Saving individual losses to {args.save_bpc}")

        # convert individual losses to bpc
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        targets = [y[0] for x, y in batches]
        targets = [t[t != -100] for t in targets]
        samples_lengths = [
            len(tokenizer.decode(t, skip_special_tokens=True)) for t in targets 
        ]

        def loss_to_bpc(loss, length):
            return loss / (log2 * length)

        assert len(article_losses) == len(
            samples_lengths
        ), f"Expected {len(samples_lengths)} losses, got {len(article_losses)} losses. Make sure to use batch_size=1."

        with open(args.save_bpc, "w") as fi:
            for loss, ln in zip(article_losses, samples_lengths):
                fi.write(f"{loss_to_bpc(loss, ln)}\n")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Causal LM evaluation")
    parser.add_argument(
        "-m",
        "--model",
        default="asafaya/kanarya-750m",
        type=str,
        help="huggingface model id or path to model directory",
    )
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument(
        "-c", "--context_size", default=None, type=int, help="Context size"
    )
    parser.add_argument("--bos_id", default=None, type=int, help="BOS token id.")
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Use wikitext-2.test.raw instead of trnews-64.test.raw",
    )
    parser.add_argument(
        "--save_bpc", type=str, default=None, help="Path to save individual losses"
    )
    parser.add_argument(
        "--contamination",
        type=str,
        default=None,
        help="Path to data for contamination test",
    )
    args = parser.parse_args()
    main(args)
