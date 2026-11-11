"""Create disjoint sample-level splits for a structure-alignment experiment."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--token', default='feature:hole')
    parser.add_argument('--test-per-class', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    yes = [s for s in samples if args.token in s.get('tokens', [])]
    no = [s for s in samples if args.token not in s.get('tokens', [])]
    count = args.test_per_class
    if count < 1 or min(len(yes), len(no)) <= count:
        raise ValueError('each class needs test samples and at least one training sample')
    rng = random.Random(args.seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    train = yes[count:] + no[count:]
    test = yes[:count] + no[:count]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'structure-train.jsonl').write_text(
        ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in train), encoding='utf-8')
    for name, group in [('train', train), ('test', test)]:
        positive = [s for s in group if args.token in s.get('tokens', [])]
        negative = [s for s in group if args.token not in s.get('tokens', [])]
        pairs = [{'base': a, 'source': b} for a in positive for b in negative]
        pairs += [{'base': b, 'source': a} for a in positive for b in negative]
        (args.out_dir / f'{name}-pairs.jsonl').write_text(
            ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in pairs), encoding='utf-8')
    print(json.dumps({'trainSamples': len(train), 'testSamples': len(test),
                      'trainPositive': len(yes) - count, 'testPositive': count}))


if __name__ == '__main__':
    main()
