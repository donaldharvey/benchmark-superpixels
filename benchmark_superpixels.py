import click
import sh
from pathlib import Path as P
import json
from csv import DictWriter
import time
import pandas

def read_log(p):
    df = pandas.read_csv(p, sep='\t', header=None)
    return {'moves': df.iloc[-1].name, 'energy': df.iloc[-1][0], 'iters': df.iloc[-1][2]}

@click.command()
@click.option('--bsds-path', type=click.Path(exists=True, file_okay=False))
@click.option('--algorithm', type=click.Choice('coarsetofine slic seeds'.split(' ')))
@click.option('--input', type=click.Path(exists=True))
@click.option('-n', '--numbers-superpixels', type=str, default='64,128,256,512,1024,2048')
@click.option('-o', '--output', type=click.Path(exists=True, file_okay=False), default='.')
def benchmark_superpixels(bsds_path, algorithm, input, numbers_superpixels, output):
    if not input:
        if ',' in numbers_superpixels:
            nums = [int(x) for x in numbers_superpixels.split(',')]
        else:
            nums = [int(numbers_superpixels)]
    else:
        nums = [int(numbers_superpixels)] if ',' not in numbers_superpixels else [100]

    for number_superpixels in nums:
        print('Running {} for n={}'.format(algorithm or 'on ' + input, number_superpixels))
        # gts = [p.parent / p.stem for p in P(bsds_path).glob('*.jpg')]
        def gts_for(gt_base):
            return gt_base.parent.glob(gt_base.stem + '_?.dat')
        if input:
            if P(input).is_dir():
                input_pngs = P(input).glob('*.png')
            else:
                input_pngs = [input]
            # name_set = set(i.with_suffix('').name for i in input_dats)
            # gts = [g for g in gts if g.name in name_set]
            args = [['--input-image', P(bsds_path) / (x.stem + '.jpg'), '--input', x] for x in input_pngs]
        else:
            args = [['--input-image', x, '--algorithm', algorithm] for x in P(bsds_path).glob("*.jpg")]

        args = [ [a + ['-g', x] for x in gts_for(a[1])] for a in args ]

        if algorithm:
            output_filename = P(output) / "bench-{}-{}.tsv".format(number_superpixels, algorithm)
        else:
            output_filename = P(output) / (P(input).stem + '.tsv')
        with output_filename.open('w') as f:
            w = DictWriter(f, delimiter='\t', fieldnames='name number_segments br2 br1 br0 asa compactness reconstruction_error ue moves energy iters'.split(' '))
            w.writeheader()
            total = len(args)
            for i, a in enumerate(args):
                log_path = a[0][3].parent / a[0][3].name.replace('.png', '.log')
                if log_path.exists():
                    log_res = read_log(str(log_path))
                else:
                    log_res = {}

                print("Running on {} ({}%)".format(a[0][1].name, round(i*100/total, 2)))
                cmds = [sh.Command('./benchmark-single')(*the_args, _bg=True) for the_args in a]
                while True:
                    if all(c.exit_code is not None for c in cmds):
                        break
                    time.sleep(0.001)

                for c in cmds:
                    res = json.loads(str(c))
                    res['name'] = a[0][1].name
                    res.update(log_res)

                    w.writerow(res)
                    f.flush()


if __name__ == '__main__':
    benchmark_superpixels()