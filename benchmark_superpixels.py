import click
import sh
from pathlib import Path as P
import json
from csv import DictWriter


@click.command()
@click.option('--bsds-path', type=click.Path(exists=True, file_okay=False))
@click.option('--algorithm', type=click.Choice('coarsetofine slic seeds'.split(' ')))
@click.option('--input', type=click.Path(exists=True))
@click.option('-n', '--number-superpixels', type=int, default=1000)
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False), default='bench-results.tsv')
def benchmark_superpixels(bsds_path, algorithm, input, number_superpixels, output):
    import ipdb; ipdb.set_trace()
    gts = [p.with_suffix('') for p in P(bsds_path).glob('*.jpg')]
    def gts_for(gt_base):
        return [i for x in gt_base.parent.glob(gt_base.stem + '_?.dat') for i in ('-g', x) ]
    if input:
        if P(input).is_dir():
            input_dats = glob.glob(P(input) / '*.dat')
        else:
            input_dats = [input]
        # name_set = set(i.with_suffix('').name for i in input_dats)
        # gts = [g for g in gts if g.name in name_set]
        args = [['--input-image', P(bsds_path) / x.stem + '.jpg', '--input', x] for x in input_dats]
    else:
        args = [['--input-image', x, '--algorithm', algorithm] for x in P(bsds_path).glob("*.jpg")]

    
    args = [a + ['--number-superpixels', number_superpixels] + gts_for(a[1]) for a in args]

    with open(output, 'w') as f:
        w = DictWriter(f, delimiter='\t', fieldnames='name number_segments br asa compactness reconstruction_error ue'.split(' '))
        w.writeheader()
        for a in args:
            res = json.loads(str(sh.Command('./benchmark-single')(*a)))
            res['name'] = a[1].name
            w.writerow(res)
            f.flush()


if __name__ == '__main__':
    benchmark_superpixels()