import argparse
from speech_isolation.infer import infer
from speech_isolation.models import IdentitySeparator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out_dir', default='out')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    model = IdentitySeparator()
    infer(model, args.input, args.out_dir, device=args.device)

if __name__ == '__main__':
    main()
