"""Provide unit tests for multivariate Spectral TE estimation."""
import pytest
import itertools as it
import numpy as np
import numpy as np
import os
from idtxl.data import Data
from idtxl.multivariate_spectral_te import MultivariateSpectralTE
from idtxl.visualise_graph import plot_spectral_result, plot_SOSO_result
import pickle
import time
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.estimators_jidt import JidtDiscreteCMI, JidtKraskovTE
from test_estimators_jidt import jpype_missing
from test_results import _get_discrete_gauss_data
from test_checkpointing import _clear_ckp
from idtxl.idtxl_utils import calculate_mi
from test_estimators_jidt import _get_gauss_data

SEED = 0

def test_spec_return_local_values():
    max_lag = 5
    data = Data(seed=SEED)
    data.generate_mute_data(500, 5)

    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'max_lag_target': 6,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    nw_0 = MultivariateTE()

    # Test all to all analysis
    result = nw_0.analyse_network(
        settings, data, targets='all', sources='all')

    source = 0
    target = 1
    spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                         'n_perm_spec': 41,
                         'n_scale': 5,
                         'wavelet': 'la8',  # or la16, mother wavelets
                         'alpha_spec': 0.05,
                         'permute_in_time_spec': True,
                         'perm_type_spec': 'block',
                         'block_size_spec': 1,
                         'perm_range_spec': int(data.n_samples/1),
                         'spectral_analysis_type': 'both',
                         'fdr_corrected': False,
                         'parallel_surr': True,
                         'surr_type': 'spectr',  # or 'iaaft'
                         'n_jobs': 6,
                         'verb_parallel': 50}
    # Run spectral TE analysis on significant source from Multivariate TE.
    spectral_analysis = MultivariateSpectralTE()



    result_spectral = spectral_analysis.analyse_network(
        spectral_settings, data, result, sources=[source], targets=[target])

    spectral_settings['local_values'] = False
    results_avg = spectral_analysis.analyse_network(
        spectral_settings, data, result, sources=[source], targets=[target])

    # Test if any sources were inferred. If not, return (this may happen
    # sometimes due to too few samples, however, a higher no. samples is not
    # feasible for a unit test).
    if result_spectral.get_single_target(target, fdr=False)['te'] is None:
        return
    if results_avg.get_single_target(target, fdr=False)['te'] is None:
        return

    lte = results.get_single_target(target, fdr=False)['te']
    n_sources = len(results.get_target_sources(target, fdr=False))
    assert type(lte) is np.ndarray, (
        'LTE estimation did not return an array of values: {0}'.format(lte))
    assert lte.shape[0] == n_sources, (
        'Wrong dim (no. sources) in LTE estimate: {0}'.format(lte.shape))
    assert lte.shape[1] == data.n_realisations_samples((0, max_lag)), (
        'Wrong dim (no. samples) in LTE estimate: {0}'.format(lte.shape))
    assert lte.shape[2] == data.n_replications, (
        'Wrong dim (no. replications) in LTE estimate: {0}'.format(lte.shape))

def test_spec_discrete_input():
    """Test multivariate TE estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    data = _get_discrete_gauss_data(covariance=covariance,
                                    n=1000,
                                    delay=1,
                                    normalise=False,
                                    seed=SEED)
    corr_expected = covariance / (
        1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,  # alphabet size of the variables analysed
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 1,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'tau_sources': 1,
        'tau_target': 1,
        'alpha': 0.05,
        'fdr_correction': False
        }
    nw = MultivariateTE()
    result = nw.analyse_single_target(settings=settings, data=data, target=1)

    source = 0
    target = 1
    spectral_settings = {'cmi_estimator': 'JidtDiscreteCMI',
                         'n_perm_spec': 41,
                         #'n_scale': 5,
                         'wavelet': 'la8',  # or la16, mother wavelets
                         'alpha_spec': 0.05,
                         'n_discrete_bins': 5,  # alphabet size of the variables analysed
                         'n_perm_max_stat': 21,
                         'n_perm_omnibus': 30,
                         'n_perm_max_seq': 30,
                         'min_lag_sources': 1,
                         'max_lag_sources': 2,
                         'max_lag_target': 1,
                         'tau_sources': 1,
                         'tau_target': 1,
                         'permute_in_time_spec': True,
                         'perm_type_spec': 'block',
                         'block_size_spec': 1,
                         'perm_range_spec': int(data.n_samples/1),
                         'spectral_analysis_type': 'both',
                         'fdr_corrected': False,
                         'parallel_surr': True,
                         'surr_type': 'spectr',  # or 'iaaft'
                         'n_jobs': 6,
                         'verb_parallel': 50}

    spectral_analysis = MultivariateSpectralTE()
    result_spectral = spectral_analysis.analyse_network(
        spectral_settings, data, result, sources=[source], targets=[target])

def test_spec_analyse_network():
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    data = Data(seed=SEED)
    data.generate_mute_data(10, 5)
    #WARNING: Number of replications is not sufficient to generate the desired number of surrogates. Permuting samples in time instead.
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'max_lag_target': 6,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    nw_0 = MultivariateTE()

    # Test all to all analysis
    results = nw_0.analyse_network(
        settings, data, targets='all', sources='all')
    targets_analysed = results.targets_analysed
    sources = np.arange(n_processes)

    source = 0
    target = 1
    spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                         'n_perm_spec': 41,
                         'n_scale': 5,
                         'wavelet': 'la8',  # or la16, mother wavelets
                         'alpha_spec': 0.1,
                         'permute_in_time_spec': True,
                         'perm_type_spec': 'block',
                         'block_size_spec': 1,
                         'perm_range_spec': int(data.n_samples/1),
                         'spectral_analysis_type': 'both',
                         'fdr_corrected': False,
                         'parallel_surr': True,
                         'surr_type': 'spectr',  # or 'iaaft'
                         'n_jobs': 6,
                         'verb_parallel': 50}

    spectral_analysis = MultivariateSpectralTE()
    result = spectral_analysis.analyse_network(
        spectral_settings, data, results, sources=[source], targets=[target])

    assert all(np.array(targets_analysed) == np.arange(n_processes)), (
                'Network analysis did not run on all targets.')
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))
    # Test analysis for subset of targets
    target_list = [1, 2, 3]
    results = nw_0.analyse_network(
        settings, data, targets=target_list, sources='all')
    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run on correct subset of targets.')
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))

    # Test analysis for subset of sources
    source_list = [1, 2, 3]
    target_list = [0, 4]
    results = nw_0.analyse_network(settings, data, targets=target_list,
                                   sources=source_list)

    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run for all targets.')
    for t in results.targets_analysed:
        assert all(results._single_target[t].sources_tested ==
                   np.array(source_list)), (
                        'Network analysis did not run on the correct subset '
                        'of sources for target {0}'.format(t))


#Implement check source set for spec required
def test_check_spec_source_set():
    """Test the method _check_source_set.

    This method sets the list of source processes from which candidates are
    taken for multivariate TE estimation.
    """
    data = Data(seed=SEED)
    data.generate_mute_data(100, 5)
    nw_0 = MultivariateSpectralTE()
    nw_0.settings = {'verbose': True}
    # Add list of sources.
    sources = [1, 2, 3]
    nw_0._check_source_set(sources, data.n_processes)
    assert nw_0.source_set == sources, 'Sources were not added correctly.'

    # Assert that initialisation fails if the target is also in the source list
    sources = [0, 1, 2, 3]
    nw_0.target = 0
    with pytest.raises(RuntimeError):
        nw_0._check_source_set(sources=[0, 1, 2, 3],
                               n_processes=data.n_processes)

    # Test if a single source, no list is added correctly.
    sources = 1
    nw_0._check_source_set(sources, data.n_processes)
    assert (type(nw_0.source_set) is list)

    # Test if 'all' is handled correctly
    nw_0.target = 0
    nw_0._check_source_set('all', data.n_processes)
    assert nw_0.source_set == [1, 2, 3, 4], 'Sources were not added correctly.'

    # Test invalid inputs.
    with pytest.raises(RuntimeError):   # sources greater than no. procs
        nw_0._check_source_set(8, data.n_processes)
    with pytest.raises(RuntimeError):  # negative value as source
        nw_0._check_source_set(-3, data.n_processes)

def test_spec_multivariate_te_init():
    """Test instance creation for MultivariateTE class."""
    # Test error on missing estimator
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    data = Data(seed=SEED)
    data.generate_mute_data(10, 5)
    #WARNING: Number of replications is not sufficient to generate the desired number of surrogates. Permuting samples in time instead.
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'max_lag_target': 6,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    nw = MultivariateTE()

    # Test all to all analysis
    results = nw.analyse_network(
        settings, data, targets='all', sources='all')

    source = 0
    target = 1
    spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                         'n_perm_spec': 41,
                         'n_scale': 5,
                         'wavelet': 'la8',  # or la16, mother wavelets
                         'alpha_spec': 0.1,
                         'permute_in_time_spec': True,
                         'perm_type_spec': 'block',
                         'block_size_spec': 1,
                         'perm_range_spec': int(data.n_samples/1),
                         'spectral_analysis_type': 'both',
                         'fdr_corrected': False,
                         'parallel_surr': True,
                         'surr_type': 'spectr',  # or 'iaaft'
                         'n_jobs': 6,
                         'verb_parallel': 50}
    # Run spectral TE analysis on significant source from Multivariate TE.
    spectral_analysis = MultivariateSpectralTE()
    result_spectral = spectral_analysis.analyse_network(
        spectral_settings, data, results, sources=[source], targets=[target])
    # Valid: max lag sources bigger than max lag target
    nw.analyse_single_target(settings=settings, data=data, target=1)

    # Valid: max lag sources smaller than max lag target
    settings['max_lag_sources'] = 3
    nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: min lag sources bigger than max lag0
    settings['min_lag_sources'] = 8
    settings['max_lag_sources'] = 7
    settings['max_lag_target'] = 5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: taus bigger than lags
    settings['min_lag_sources'] = 2
    settings['max_lag_sources'] = 4
    settings['max_lag_target'] = 5
    settings['tau_sources'] = 10
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['tau_sources'] = 1
    settings['tau_target'] = 10
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: negative lags or taus
    settings['min_lag_sources'] = 1
    settings['max_lag_target'] = 5
    settings['max_lag_sources'] = -7
    settings['tau_target'] = 1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['max_lag_sources'] = 7
    settings['min_lag_sources'] = -4
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['min_lag_sources'] = 4
    settings['max_lag_target'] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['max_lag_target'] = 5
    settings['tau_sources'] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['tau_sources'] = 1
    settings['tau_target'] = -1
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)

    # Invalid: lags or taus are no integers
    settings['tau_target'] = 1
    settings['min_lag_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['min_lag_sources'] = 1
    settings['max_lag_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['max_lag_sources'] = 7
    settings['tau_sources'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['tau_sources'] = 1
    settings['tau_target'] = 1.5
    with pytest.raises(RuntimeError):
        nw.analyse_single_target(settings=settings, data=data, target=1)
    settings['tau_target'] = 1

    # Invalid: sources or target is no int
    with pytest.raises(RuntimeError):  # no int
        nw.analyse_single_target(settings=settings, data=data, target=1.5)
    with pytest.raises(RuntimeError):  # negative
        nw.analyse_single_target(settings=settings, data=data, target=-1)
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=10)
    with pytest.raises(RuntimeError):  # wrong type
        nw.analyse_single_target(settings=settings, data=data, target={})
    with pytest.raises(RuntimeError):  # negative
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=-1)
    with pytest.raises(RuntimeError):   # negative
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=[-1])
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=20)
    with pytest.raises(RuntimeError):  # not in data
        nw.analyse_single_target(settings=settings, data=data, target=0,
                                 sources=[20])

def test_spec_multivariate_te_one_realisation_per_replication():
    """Test boundary case of one realisation per replication."""
    # Create a data set where one pattern fits into the time series exactly
    # once, this way, we get one realisation per replication for each variable.
    # This is easyer to assert/verify later. We also test data.get_realisations
    # this way.
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 5,
        'min_lag_sources': 1,
        'max_lag_target': 5,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    target = 0
    data = Data(normalise=False, seed=SEED)
    n_repl = 10
    n_procs = 2

    #network_analysis = MultivariateTE()
    #result = network_analysis.analyse_network(settings=settings, data=data)

    n_points = n_procs * (settings['max_lag_sources'] + 1) * n_repl
    data.set_data(np.arange(n_points).reshape(
        n_procs, settings['max_lag_sources'] + 1, n_repl), 'psr')
    nw_0 = MultivariateTE()
    nw_0._initialise(settings, data, 'all', target)
    assert (not nw_0.selected_vars_full)
    assert (not nw_0.selected_vars_sources)
    assert (not nw_0.selected_vars_target)
    assert ((nw_0._replication_index == np.arange(n_repl)).all())
    assert (nw_0._current_value == (target, max(
           settings['max_lag_sources'], settings['max_lag_target'])))
    assert (nw_0._current_value_realisations[:, 0] ==
            data.data[target, -1, :]).all()

@jpype_missing
def test_spec_faes_method():
    """Check if the Faes method is working."""

    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 5,
        'min_lag_sources': 3,
        'max_lag_target': 7,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    nw = MultivariateTE()

    data = Data(seed=SEED)
    data.generate_mute_data()
    sources = [1, 2, 3]
    target = 0

    # Test all to all analysis
    results = nw.analyse_network(
        settings, data, targets='all', sources='all')

    spec_settings = {'cmi_estimator': 'JidtKraskovCMI',
                'add_conditionals': 'faes',
                'max_lag_sources': 5,
                'min_lag_sources': 3,
                'max_lag_target': 7,
                'tau_sources': 1,
                'tau_target': 1,
                'n_perm_max_stat': 200,
                'n_perm_min_stat':  200,
                'n_perm_omnibus':  200,
                'n_perm_max_seq': 200,
                'alpha': 0.05,
                'fdr_correction': False}

    nw_1 = MultivariateSpectralTE()#MultivariateTE()

    nw_1._initialise(settings=spec_settings, data=data, results=results, sources=sources, target=target)
    assert (nw_1._selected_vars_sources ==
            [i for i in it.product(sources, [nw_1.current_value[1]])]), (
                'Did not add correct additional conditioning vars.')

def test_spec_add_conditional_manually():
    """Enforce the conditioning on additional variables."""
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 3,
                'max_lag_target': 7,
                'tau_sources': 1,
                'tau_target': 1,
                'n_perm_max_stat': 200,
                'n_perm_min_stat':  200,
                'n_perm_omnibus':  200,
                'n_perm_max_seq': 200,
                'alpha': 0.05,
                'fdr_correction': False}
    nw = MultivariateSpectralTE()
    data = Data(seed=SEED)
    data.generate_mute_data()

    # Add a conditional with a lag bigger than the max_lag requested above
    settings['add_conditionals'] = (8, 0)
    with pytest.raises(IndexError):
        nw._initialise(settings, data, sources=[1, 2], target=0)

    # Add valid conditionals and test if they were added
    settings['add_conditionals'] = [(0, 1), (1, 3)]
    nw._initialise(settings=settings, data=data, target=0, sources=[1, 2])
    # Get list of conditionals after intialisation and convert absolute samples
    # back to lags for comparison.
    cond_list = nw._idx_to_lag(nw.selected_vars_full)
    assert settings['add_conditionals'][0] in cond_list, (
        'First enforced conditional is missing from results.')
    assert settings['add_conditionals'][1] in cond_list, (
        'Second enforced conditional is missing from results.')


def test_spec_define_candidates():
    """Test candidate definition from a list of procs and a list of samples."""
    target = 1
    tau_target = 3
    max_lag_target = 10
    current_val = (target, 10)
    procs = [target]
    samples = np.arange(current_val[1] - 1, current_val[1] - max_lag_target,
                        -tau_target)
    # Test if candidates that are added manually to the conditioning set are
    # removed from the candidate set.
    nw = MultivariateSpectralTE()
    nw.current_value = current_val
    settings = [
        {'add_conditionals': None},
        {'add_conditionals': (2, 3)},
        {'add_conditionals': [(2, 3), (4, 1)]},
        {'add_conditionals': [(1, 9)]},
        {'add_conditionals': [(1, 9), (2, 3), (4, 1)]}]
    for s in settings:
        nw.settings = s
        candidates = nw._define_candidates(procs, samples)
        assert (1, 9) in candidates, 'Sample missing from candidates: (1, 9).'
        assert (1, 6) in candidates, 'Sample missing from candidates: (1, 6).'
        assert (1, 3) in candidates, 'Sample missing from candidates: (1, 3).'
        if s['add_conditionals'] is not None:
            if type(s['add_conditionals']) is tuple:
                cond_ind = nw._lag_to_idx([s['add_conditionals']])
            else:
                cond_ind = nw._lag_to_idx(s['add_conditionals'])
            for c in cond_ind:
                assert c not in candidates, (
                    'Sample added erronously to candidates: {}.'.format(c))

#test for running spectral analyis on other data than the data for results
def test_spec_data():
        data = Data(seed=SEED)
        data.generate_mute_data()
        source = 0
        target = 1

        network_analysis = MultivariateTE()
        settings = {
            'cmi_estimator': 'JidtKraskovCMI',
            'max_lag_sources': 3,
            'min_lag_sources': 1,
            'max_lag_target': 6,
            'tau_sources': 1,
            'tau_target': 1,
            'n_perm_max_stat': 200,
            'n_perm_min_stat':  200,
            'n_perm_omnibus':  200,
            'n_perm_max_seq': 200,
            'alpha': 0.05,
            'fdr_correction': False
        }
        data = Data(seed=SEED)
        data.generate_mute_data(10, 5)
        result = network_analysis.analyse_network(settings=settings, data=data)

        data2 = Data(seed=1)
        data2.generate_mute_data()

        spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                             'n_perm_spec': 41,
                             'n_scale': 5,
                             'wavelet': 'la8',  # or la16, mother wavelets
                             'alpha_spec': 0.1,
                             'permute_in_time_spec': True,
                             'perm_type_spec': 'block',
                             'block_size_spec': 1,
                             'perm_range_spec': int(data.n_samples/1),
                             'spectral_analysis_type': 'both',
                             'fdr_corrected': False,
                             'parallel_surr': True,
                             'surr_type': 'spectr',  # or 'iaaft'
                             'n_jobs': 6,
                             'verb_parallel': 50}
        # Run spectral TE analysis on significant source from Multivariate TE.
        spectral_analysis = MultivariateSpectralTE()
        result_spectral = spectral_analysis.analyse_network(
            spectral_settings, data2, result, sources=[source], targets=[target])

def test_spec_checkpoint():
    """Test method for full network analysis."""
    n_processes = 5  # the MuTE network has 5 nodes
    data = Data(seed=SEED)
    data.generate_mute_data(10, 5)
    filename_ckp = './my_checkpoint'
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'max_lag_target': 6,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'write_ckp': True,
        'filename_ckp': filename_ckp}

    nw_0 = MultivariateTE()

    # Test all to all analysis
    result = nw_0.analyse_network(
        settings, data, targets='all', sources='all')
    source = 0
    target = 1
    spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                         'n_perm_spec': 41,
                         'n_scale': 5,
                         'wavelet': 'la8',  # or la16, mother wavelets
                         'alpha_spec': 0.05,
                         'permute_in_time_spec': True,
                         'perm_type_spec': 'block',
                         'block_size_spec': 1,
                         'perm_range_spec': int(data.n_samples/1),
                         'spectral_analysis_type': 'both',
                         'fdr_corrected': False,
                         'parallel_surr': True,
                         'surr_type': 'spectr',  # or 'iaaft'
                         'n_jobs': 6,
                         'verb_parallel': 50,
                         'write_ckp': True,
                         'filename_ckp': filename_ckp}
    # Run spectral TE analysis on significant source from Multivariate TE.
    spectral_analysis = MultivariateSpectralTE()
    results = spectral_analysis.analyse_network(
        spectral_settings, data, result, sources=[source], targets=[target])




    targets_analysed = results.targets_analysed
    sources = np.arange(n_processes)
    assert all(np.array(targets_analysed) == np.arange(n_processes)), (
                'Network analysis did not run on all targets.')
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))
    # Test analysis for subset of targets
    target_list = [1, 2, 3]
    results = nw_0.analyse_network(
        settings, data, targets=target_list, sources='all')
    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run on correct subset of targets.')
    for t in results.targets_analysed:
        s = np.array(list(set(sources) - set([t])))
        assert all(np.array(results._single_target[t].sources_tested) == s), (
                    'Network analysis did not run on all sources for target '
                    '{0}'. format(t))

    # Test analysis for subset of sources
    source_list = [1, 2, 3]
    target_list = [0, 4]
    results = nw_0.analyse_network(settings, data, targets=target_list,
                                   sources=source_list)

    targets_analysed = results.targets_analysed
    assert all(np.array(targets_analysed) == np.array(target_list)), (
                'Network analysis did not run for all targets.')
    for t in results.targets_analysed:
        assert all(results._single_target[t].sources_tested ==
                   np.array(source_list)), (
                        'Network analysis did not run on the correct subset '
                        'of sources for target {0}'.format(t))

    _clear_ckp(filename_ckp)

if __name__ == '__main__':
    test_spec_return_local_values()
    
    test_spec_discrete_input() # Maximum of var2 is larger than the alphabet size
    test_spec_analyse_network() # AssertionError: scale (5) must be smaller or equal to max_scale (3).
    test_check_spec_source_set() # Implement check_source_set
    test_spec_multivariate_te_init() # AssertionError: scale (5) must be smaller or equal to max_scale (3).
    test_spec_multivariate_te_one_realisation_per_replication()
    test_spec_faes_method() # RuntimeError: Conflicting entries in spectral TE and network inference settings.
    test_spec_add_conditional_manually()
    test_spec_data()
    test_spec_define_candidates() #Works
    test_spec_checkpoint() # AssertionError: scale (5) must be smaller or equal to max_scale (3).
    
