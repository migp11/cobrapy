# -*- coding: utf-8 -*-

import sys
import multiprocessing
import logging
import optlang
import pandas as pd
from warnings import warn
from itertools import product
from functools import partial
from builtins import (map, dict)
from future.utils import raise_
from cobra.manipulation.delete import find_gene_knockout_reactions
import cobra.util.solver as sutil


try:
    import scipy
except ImportError:
    moma = None
else:
    from cobra.flux_analysis import moma

LOGGER = logging.getLogger(__name__)


def _reactions_knockouts_with_restore(model, reactions):
    with model:
        for reaction in reactions:
            reaction.knock_out()
        growth = _get_growth(model)
    return ([r.id for r in reactions], growth, model.solver.status)


def _get_growth(model):
    try:
        if 'moma_old_objective' in model.solver.variables:
            model.slim_optimize()
            growth = model.solver.variables.moma_old_objective.primal
        else:
            growth = model.slim_optimize()
    except optlang.exceptions.SolverError:
        growth = float('nan')
    return growth


def _reaction_deletion(model, ids):
    return _reactions_knockouts_with_restore(
        model,
        [model.reactions.get_by_id(r_id) for r_id in ids]
    )


def _gene_deletion(model, ids):
    all_reactions = []
    for g_id in ids:
        all_reactions.extend(
            find_gene_knockout_reactions(
                model, (model.genes.get_by_id(g_id),)
            )
        )
    _, growth, status = _reactions_knockouts_with_restore(model, all_reactions)
    return (ids, growth, status)


def _reaction_deletion_worker(ids):
    global _model
    return _reaction_deletion(_model, ids)


def _gene_deletion_worker(ids):
    global _model
    return _gene_deletion(_model, ids)


def _init_worker(model):
    global _model
    _model = model


def _multi_deletion(cobra_model, entity, element_lists, method="fba",
                    number_of_processes=None, solver=None):
    """
    Helper function that provides the common interface for sequential
    knockouts

    Parameters
    ----------
    cobra_model : cobra.Model
        The metabolic model to perform deletions in.

    entity : 'gene' or 'reaction'
        The entity to knockout (`cobra.Gene` or `cobra.Reaction`)

    element_lists : list
        List of iterables `cobra.Reaction`s or `cobra.Gene`s (or their IDs)
        to be deleted.

    method: {"fba", "moma"}, optional
        Procedure used to predict the growth rate.

    number_of_processes : int, optional
        Number of parallel processes to run. Can speed up the computations
        if number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs

    solver: str, optional
        This must be a QP-capable solver for MOMA. If left unspecified,
        a suitable solver will be automatically chosen.

    Returns
    -------
    pandas.DataFrame
        A sparse representation of all combinations of entities
        deletions. The columns are ['ids', 'growth', 'status'], where
        ids : frozenset([str])
            all the knockouts performed for one case
        growth : float
            growth rate of optimized model
        status : str
            solver status
    """
    with cobra_model as model:
        try:
            (legacy, solver) = sutil.choose_solver(model, solver,
                                                   qp=(method == "moma"))
        except sutil.SolverNotFound:
            if method == "moma":
                warn(
                    "Cannot use MOMA since no QP-capable solver was found. "
                    "Falling back to FBA.")
                (legacy, solver) = sutil.choose_solver(model, solver)
            else:
                (err_type, err_val, err_tb) = sys.exc_info()
                raise_(err_type, err_val, err_tb)  # reraise for Python2&3
        if legacy:
            raise ValueError(
                "Legacy solvers are not supported any longer. "
                "Please use one of the optlang solver interfaces instead.")

        if number_of_processes is None:
            try:
                num_cpu = multiprocessing.cpu_count()
            except NotImplementedError:
                warn("Number of cores could not be detected - assuming 1.")
                num_cpu = 1
        else:
            num_cpu = number_of_processes

        if 'moma' in method:
            moma.add_moma(model, linear='linear' in method)

        args = set([frozenset(comb) for comb in product(*element_lists)])
        num_cpu = max(num_cpu, len(args))

        def extract_knockout_results(results):
            result = pd.DataFrame([
                [frozenset(ids), growth, status]
                for (ids, growth, status) in results
            ], columns=['ids', 'growth', 'status'])
            result = result.set_index('ids')
            return result

        if num_cpu > 1:
            WORKER_FUNCTIONS = dict(
                gene=_gene_deletion_worker,
                reaction=_reaction_deletion_worker
            )
            chunk_size = len(args) // num_cpu
            pool = multiprocessing.Pool(
                num_cpu, initializer=_init_worker, initargs=(model,)
            )
            results = extract_knockout_results(pool.imap_unordered(
                WORKER_FUNCTIONS[entity],
                args,
                chunksize=chunk_size
            ))
            pool.close()
        else:
            WORKER_FUNCTIONS = dict(
                gene=_gene_deletion,
                reaction=_reaction_deletion
            )
            results = extract_knockout_results(map(
                partial(WORKER_FUNCTIONS[entity], model), args
            ))
        return results


def _entities_ids(entities):
    try:
        return [e.id for e in entities]
    except AttributeError:
        return list(entities)


def _element_lists(entities, *ids):
    lists = list(ids)
    if lists[0] is None:
        lists[0] = entities
    result = [_entities_ids(lists[0])]
    for l in lists[1:]:
        if l is None:
            result.append(result[-1])
        else:
            result.append(_entities_ids(l))
    return result


def single_reaction_deletion(model, reaction_list=None, **kwargs):
    """Sequentially knocks out each reaction from a given reaction list.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to perform deletions in.

    reaction_list : iterable
        Iterable `cobra.Reaction`s to be deleted. If not passed,
        all the reactions from the model are used.

    method: {"fba", "moma"}, optional
        Procedure used to predict the growth rate.

    number_of_processes : int, optional
        Number of parallel processes to run. Can speed up the computations
        if number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs

    solver: str, optional
        This must be a QP-capable solver for MOMA. If left unspecified,
        a suitable solver will be automatically chosen.

    Returns
    -------
    pandas.DataFrame
        A sparse representation of all combinations of entities
        deletions. The columns are ['ids', 'growth', 'status'], where
        ids : frozenset([str])
            all the knockouts performed for one case
        growth : float
            growth rate of optimized model
        status : str
            solver status
    """
    return _multi_deletion(
        model,
        'reaction',
        element_lists=_element_lists(model.reactions, reaction_list),
        **kwargs
    )


def single_gene_deletion(model, gene_list=None, **kwargs):
    """Sequentially knocks out each gene from a given gene list.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to perform deletions in.

    gene_list : iterable
        Iterable `cobra.Gene`s to be deleted. If not passed,
        all the genes from the model are used.

    method: {"fba", "moma"}, optional
        Procedure used to predict the growth rate.

    number_of_processes : int, optional
        Number of parallel processes to run. Can speed up the computations
        if number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs

    solver: str, optional
        This must be a QP-capable solver for MOMA. If left unspecified,
        a suitable solver will be automatically chosen.

    Returns
    -------
    pandas.DataFrame
        A sparse representation of all combinations of entities
        deletions. The columns are ['ids', 'growth', 'status'], where
        ids : frozenset([str])
            all the knockouts performed for one case
        growth : float
            growth rate of optimized model
        status : str
            solver status
    """
    return _multi_deletion(
        model,
        'gene',
        element_lists=_element_lists(model.genes, gene_list),
        **kwargs
    )


def double_reaction_deletion(model,
                             reaction_list1=None, reaction_list2=None,
                             **kwargs):
    """
    Sequentially delete pairs of reactions and record the objective value.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to perform deletions in.

    reaction_list1 : iterable, optional
        First iterable of `cobra.Reaction`s to be deleted. If not passed,
        all the reactions from the model are used.

    reaction_list2 : iterable, optional
        Second iterable of `cobra.Reaction`s to be deleted. If not passed,
        all the reactions from the model are used. The product of two
        reactions lists will be used to perform the knockouts.

    method: {"fba", "moma"}, optional
        Procedure used to predict the growth rate.

    number_of_processes : int, optional
        Number of parallel processes to run. Can speed up the computations
        if number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs

    solver: str, optional
        This must be a QP-capable solver for MOMA. If left unspecified,
        a suitable solver will be automatically chosen.

    Returns
    -------
    pandas.DataFrame
        A sparse representation of all combinations of entities
        deletions. The columns are ['ids', 'growth', 'status'], where
        ids : frozenset([str])
            all the knockouts performed for one case
        growth : float
            growth rate of optimized model
        status : str
            solver status
    """

    reaction_list1, reaction_list2 = _element_lists(model.reactions,
                                                    reaction_list1,
                                                    reaction_list2)
    return _multi_deletion(model, 'reaction',
                           element_lists=[reaction_list1, reaction_list2],
                           **kwargs)


def double_gene_deletion(model, gene_list1=None, gene_list2=None, **kwargs):
    """Sequentially knocks out pairs of genes in a model

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to perform deletions in.

    gene_list1 : iterable, optional
        First iterable of `cobra.Gene`s to be deleted. If not passed,
        all the genes from the model are used.

    gene_list2 : iterable, optional
        Second iterable of `cobra.Gene`s to be deleted. If not passed,
        all the genes from the model are used. The product of two
        genes lists will be used to perform the knockouts.

    method: {"fba", "moma"}, optional
        Procedure used to predict the growth rate.

    number_of_processes : int, optional
        Number of parallel processes to run. Can speed up the computations
        if number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs

    solver: str, optional
        This must be a QP-capable solver for MOMA. If left unspecified,
        a suitable solver will be automatically chosen.

    Returns
    -------
    pandas.DataFrame
        A sparse representation of all combinations of entities
        deletions. The columns are ['ids', 'growth', 'status'], where
        ids : frozenset([str])
            all the knockouts performed for one case
        growth : float
            growth rate of optimized model
        status : str
            solver status
    """

    gene_list1, gene_list2 = _element_lists(model.genes, gene_list1,
                                            gene_list2)
    return _multi_deletion(model, 'gene',
                           element_lists=[gene_list1, gene_list2], **kwargs)


def double_deletion(cobra_model, element_list_1=None, element_list_2=None,
                    element_type='gene', **kwargs):
    """Wrapper for double_gene_deletion and double_reaction_deletion

    .. deprecated :: 0.4
        Use double_reaction_deletion and double_gene_deletion
    """
    warn(
        "deprecated - use single_reaction_deletion and single_gene_deletion")
    if element_type == "reaction":
        return double_reaction_deletion(cobra_model, element_list_1,
                                        element_list_2, **kwargs)
    elif element_type == "gene":
        return double_gene_deletion(cobra_model, element_list_1,
                                    element_list_2, **kwargs)
    else:
        raise Exception("unknown element type")
