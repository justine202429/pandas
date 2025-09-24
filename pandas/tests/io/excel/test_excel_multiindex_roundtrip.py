import pandas as pd
import numpy as np


def test_multiindex_excel_roundtrip(tmp_path):
    df = (pd.DataFrame({'a': list('ABBAAAB'),
                        'b': [-1, 1, 1, -2, float('nan'), 3, -4]})
          .assign(b_bin=lambda x: pd.cut(x.b, bins=[-float('inf'), 0, float('inf')]))
          .groupby(['b_bin', 'a'], as_index=False, observed=True, dropna=False)
          .agg(b_sum=('b', 'sum'), b_prod=('b', 'prod'))
          .pivot(index='a', columns='b_bin', values=['b_sum', 'b_prod'])
          )

    fname = tmp_path / "test.xlsx"
    df.to_excel(fname)

    df2 = pd.read_excel(fname, header=[0,1], index_col=0)

    # normalize expected columns to strings for comparison
    def normalize_cols(cols):
        out = []
        for top, sub in cols:
            if pd.isna(sub):
                sub_s = 'nan'
            else:
                sub_s = str(sub)
            out.append((top, sub_s))
        return out

    assert normalize_cols(df.columns) == normalize_cols(df2.columns)
    # compare values with NaNs treated equal
    assert df.fillna('<<NA>>').to_numpy().tolist() == df2.fillna('<<NA>>').to_numpy().tolist()
