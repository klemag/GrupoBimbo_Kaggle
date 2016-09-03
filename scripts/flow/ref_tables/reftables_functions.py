import pandas as pd
import re
import numpy as np
import sys
sys.path.append('../../flow')
import parameters as p

### Generic Functions

# Load all weeks of data with correct types, and create new features
# New features include Price per unit (ppu), Difference and Multiplied features
# New features will be added to the column_feat list
def load_weeks(column_id, column_feat, types):

    # Load data
    tab = pd.read_csv(p.DIR_ROOT + '/inputs/train.csv',
                      usecols=column_id + column_feat, dtype=types)

    # Create new features: Venta and Dev price per unit
    for x in [('Venta', 'hoy'), ('Dev', 'proxima')]:
        ppu = x[0] + '_ppu'
        units = x[0] + '_uni_' + x[1]
        price = x[0] + '_' + x[1]
        tab.loc[tab[units] != 0, ppu] = tab.loc[tab[units]
                                                != 0, price] / tab.loc[tab[units] != 0, units]
        tab.loc[tab[units] == 0, ppu] = 0
        column_feat.append(ppu)

    # Feature containing the difference of ppu/uni between expected next week (proxima)
    # and that week (hoy)
    tab['Diff_ppu_prox_hoy'] = tab.Dev_ppu - tab.Venta_ppu
    tab['Diff_uni_dem_hoy'] = tab.Demanda_uni_equil - tab.Venta_uni_hoy
    tab['Diff_uni_prox_dem'] = tab.Dev_uni_proxima - tab.Demanda_uni_equil
    tab['Mult_uni_dem_hoy'] = tab.Demanda_uni_equil * tab.Venta_uni_hoy
    column_feat += ['Diff_ppu_prox_hoy', 'Diff_uni_dem_hoy',
                 'Diff_uni_prox_dem', 'Mult_uni_dem_hoy']

    print('\nFeatures to use for aggregation:\n{}\n'.format(column_feat))

    ## Remove outliers
    for col in column_feat:

        # Remove outliers
        if p.RemoveOutliersAggFeat:
            mean_col = tab[col].mean()
            if mean_col != 0:
                 outliers = len(
                     tab.loc[abs(tab[col]) > p.OutliersThreshold * abs(mean_col)])
                 tab.loc[(abs(tab[col]) > p.OutliersThreshold * abs(mean_col))
                         & (tab[col] < 0), col] = - p.OutliersThreshold * abs(mean_col)
                 tab.loc[(abs(tab[col]) > p.OutliersThreshold * abs(mean_col))
                         & (tab[col] > 0), col] = p.OutliersThreshold * abs(mean_col)
            else:
                 outliers = 0

            print('{}: Mean={} | Outliers={}'.format(col, mean_col,outliers))

    return tab


def get_new_features(groupby_dict, feat_to_groupby, ref_table, feat_to_agg):
    print('\nBuild aggregate features')
    new_feat = pd.DataFrame()

    # For each feature to groupby, build a dictionary
    # The groupby tables are given by the input "groupby_dict"
    # Each dictionary contains a table for each combination Column_to_aggregate*Type_of_aggregation
    for gpby in feat_to_groupby:

        dict_type = {}
        # Foreach column to aggregate, use different type of agg
        for col in feat_to_agg:

            if 'mean' in p.AggFeatures_types:
                dict_type[col + '_mean'] = groupby_dict[gpby].mean().loc[:, [col]
                                                               ].to_dict()[col]
            if 'max' in p.AggFeatures_types:
                dict_type[col + '_max'] = groupby_dict[gpby].max().loc[:, [col]
                                                             ].to_dict()[col]

            if 'min' in p.AggFeatures_types:
                dict_type[col + '_min'] = groupby_dict[gpby].min().loc[:, [col]
                                           ].to_dict()[col]

            if 'med' in p.AggFeatures_types:
                dict_type[col + '_med'] = groupby_dict[gpby].median().loc[:, [col]
                                          ].to_dict()[col]

            if 'std' in p.AggFeatures_types:
                avg_std = groupby_dict[gpby].std().loc[:, [col]].mean()
                dict_type[col + '_std'] = groupby_dict[gpby].std().loc[:, [col]
                                                          ].fillna(avg_std).to_dict()[col]

        # Create the new columns and add them to the new_feat DataFrame
        for key in dict_type.keys():

            name = gpby + '_' + key

            new_feat[name] = ref_table[gpby].map(dict_type[key])
            m = new_feat.loc[~new_feat[name].isnull(), name].mean()
            new_feat.loc[new_feat[name].isnull(), name] = m

            # Normalization
            if p.NormalizeAggFeat:
                 mean_col = new_feat[name].mean()
                 max_col = new_feat[name].max()
                 min_col = new_feat[name].min()
                 diff_col = float(max_col - min_col)
                 # If column is constant, ie min==max, then the normalized column is equal to 0
                 if diff_col == 0:
                     new_feat.loc[:, name] = 0
                 else:
                     adj_mean = mean_col / diff_col
                     new_feat.loc[:, name] = (new_feat.loc[:, name] / diff_col - adj_mean).astype(np.float16)

                 print('{}: mean: {}, max: {}, min: {}'.format(name, mean_col, max_col, min_col))

                 # Multiply by the MaxValue wanted and convert to integer
                 # To save memory
                 new_feat[name] = p.MaxValueGenerated * new_feat[name]
                 new_feat[name] = new_feat[name].astype(int)

        del dict_type

    return new_feat

### Specific to products ###

# Get Brands from product table
def get_brand(s):

    # Regexp to find a Brand name
    if re.search(r" [A-Z]{1,4} [0-9]+$", s):
        return re.search(r" [A-Z]{1,4} [0-9]+$", s).group().split(' ')[1]
    else:
        return np.NaN

# Extract product names and keep them in a dictionary, with their frequency
# Then, for a given string, we will only keep the most used word as the Product name
def build_dict_products(unique_prods):

    # Build dict of products
    dict_prods = {}
    for x in unique_prods:
        for y in x.split(' '):
            # Remove irrelevant words
            # Numbers
            if re.search(r"[0-9]+", y):
                break
            # Short capitalized words
            elif re.search(r"[A-Z]{2,4}", y):
                break
            # Flavours
            elif y in ['Naranja', 'Limon', 'Fresa']:
                break
            # Too short words
            elif len(y) < 3:
                break

            # Remove plurals (s) and masculin (o) words to have only single feminine
            if re.search(r"[s]$", y):
                y = y[:-1]

            if re.search(r"[o]$", y):
                y = y[:-1] + 'a'

            # Finally add the word to the dict
            if y not in dict_prods.keys():
                dict_prods[y] = 1
            else:
                dict_prods[y] += 1

    return dict_prods

# Get product name by keeping the most used word only
def get_prod(s, dict_prods):

    score = {}

    # Some pre-processing first
    for y in s.split(' '):

        # Remove plurals and masculine words to have only single feminine
        score[y] = dict_prods.get(y, 0)
        if re.search(r"[s]$", y):
            y = y[:-1]
            score[y] = dict_prods.get(y, 0)
        if re.search(r"[o]$", y):
            y = y[:-1] + 'a'
            score[y] = dict_prods.get(y, 0)

    # Initialize best score and word
    max_word = s.split(' ')[0]
    max_score = 0

    # Get the word with the best score
    for y in score.keys():
        if score[y] > max_score:
            max_score = score[y]
            max_word = y

    return max_word

# Get weigth of the product
def get_weight(s):

    # Look for grams and kg
    if re.search(r"[ ]?[0-9]{1,4}[Kk]?[g][ ]?", s):
        we = re.search(r"[]?[0-9]{1,4}[Kk]?[g][ ]?",
                       s).group().replace(' ', '').replace('g', '')

        # convert kg to grams
        if 'K' in we or 'k' in we:
            return float(we.replace('K', '').replace('k', '')) * 1000
        return float(re.search(r"[ ]?[0-9]{1,4}[Kk]?[g][ ]?", s).group().replace(' ', '').replace('g', ''))

    # Look for ml for liquids
    # Assume N(ml)=n(g)
    elif re.search(r" [0-9]{1,4}[ ]?ml ", s):
        return float(re.search(r" [0-9]{1,4}[ ]?ml ", s).group().replace('ml ', ''))

    # Some products will have the number of pieces. In this case multiply by an arbitrary weight to get some
    # approximation -- and be able to compare products with more or less pieces
    elif re.search(r" [0-9]{1,4}[ ]?[Pp] ", s):
        return float(
            re.search(r" [0-9]{1,4}[ ]?[Pp] ", s).group().replace('P', '').replace('p', '').replace(' ', '')) * 10

    else:
        return np.NaN


# Specific for cliente

# Get Cliente Group
# Same process as for products, we build a dict of frequencies and get the most used word
def build_dict_cliente(unique_cliente):

    # Build dict cliente
    dict_cliente = {}
    for x in unique_cliente:

        for y in x.split(' '):

            # Remove irrelevant words
            if re.search(r"[0-9]+", y):
                break
            elif y == '':
                break
            elif y in ['DE', 'Y', 'EL', 'LA', 'R', 'LAS', 'LOS', 'DEL', 'SA', 'I', 'II', 'M', 'S', 'A', 'C']:
                break

            if y not in dict_cliente.keys():
                dict_cliente[y] = 1
            else:
                dict_cliente[y] += 1

    return dict_cliente


def get_cliente_group(s, dict_cliente):

    score = {}
    for y in s.split(' '):
        # Remove plurals and masculine words to have only single feminine
        score[y] = dict_cliente.get(y, 0)

    # Initialize best scores
    max_word = s.split(' ')[0]
    max_score = 0

    for y in score.keys():
        if score[y] > max_score:
            max_score = score[y]
            max_word = y

    return max_word