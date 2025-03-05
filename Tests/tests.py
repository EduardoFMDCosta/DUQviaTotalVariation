import unittest
import torch

import grid_generation as grid
import probability_mass_computation as proba
import bounds as tv
from dynamics import LinearDynamics

from scipy.stats import norm

# Import your test cases

class Tests(unittest.TestCase):

    def test_linear_dynamics(self):
        A = torch.Tensor(
            [
                [0.84, 0.1],
                [0.05, 0.72]
            ])

        f = LinearDynamics(A)
        points = torch.Tensor([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]])
        propagated = f(points)

        expected_result = torch.Tensor([[0, 0], [0.84, 0.05], [0.1, 0.72], [0.94, 0.77], [1.78, 0.82]])

        torch.testing.assert_close(propagated, expected_result)

    def test_high_prob_region_generation(self):

        points = torch.Tensor([[0.5, 0.7, 1.6], [1.0, 2.6, 8.9], [0.1, 0.4, 0.8], [1.6, 2.8, 3.4]])
        hpr = grid.identify_high_prob_region(points)

        expected_result = torch.Tensor([[0.1, 0.4, 0.8], [1.6, 2.8, 8.9]])

        torch.testing.assert_close(hpr, expected_result)

    def test_get_vertices(self):

        regions = torch.Tensor([[[0.1, 0.4, 0.8], [1.6, 2.8, 8.9]]])

        vertices = grid.get_vertices(regions)

        expected_result = torch.Tensor([[[0.1, 0.4, 0.8],
                                        [0.1, 0.4, 8.9],
                                        [0.1, 2.8, 0.8],
                                        [0.1, 2.8, 8.9],
                                        [1.6, 0.4, 0.8],
                                        [1.6, 0.4, 8.9],
                                        [1.6, 2.8, 0.8],
                                        [1.6, 2.8, 8.9]]])

        torch.testing.assert_close(vertices, expected_result)

    def test_get_all_vertices_at_once(self):

        regions = torch.Tensor([[[0.00, 0.00], [0.50, 0.50]],
                                [[0.50, 0.00], [1.00, 0.50]]
                                ])

        vertices = grid.get_vertices(regions)

        expected_result = torch.Tensor([
            [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
            [[0.5, 0.0], [0.5, 0.5], [1.0, 0.0], [1.0, 0.5]]])

        torch.testing.assert_close(vertices, expected_result)


    # def test_region_generation(self):
    #
    #     hpr = torch.Tensor([[0.0, 0.0], [1.0, 1.0]])
    #     samples = torch.Tensor([[0.1, 0.1],
    #                             [0.1, 0.3],
    #                             [0.1, 0.7],
    #                             [0.1, 0.8],
    #                             [0.3, 0.1],
    #                             [0.3, 0.3],
    #                             [0.3, 0.7],
    #                             [0.3, 0.8],
    #                             [0.7, 0.1],
    #                             [0.7, 0.3],
    #                             [0.7, 0.7],
    #                             [0.7, 0.8],
    #                             [0.8, 0.1],
    #                             [0.8, 0.3],
    #                             [0.8, 0.7],
    #                             [0.8, 0.8],
    #                             ])
    #
    #     regions = grid.create_regions(hpr, samples, 0.1, 0.1)
    #
    #     expected_result = torch.Tensor([[[0.00, 0.00],[0.25, 0.25]],
    #                                     [[0.25, 0.00], [0.50, 0.25]],
    #                                     [[0.00, 0.25], [0.25, 0.50]],
    #                                     [[0.25, 0.25], [0.50, 0.50]],
    #                                     [[0.50, 0.00], [0.75, 0.25]],
    #                                     [[0.75, 0.00], [1.00, 0.25]],
    #                                     [[0.50, 0.25], [0.75, 0.50]],
    #                                     [[0.75, 0.25], [1.00, 0.50]],
    #                                     [[0.00, 0.50], [0.25, 0.75]],
    #                                     [[0.25, 0.50], [0.50, 0.75]],
    #                                     [[0.00, 0.75], [0.25, 1.00]],
    #                                     [[0.25, 0.75], [0.50, 1.00]],
    #                                     [[0.50, 0.50], [0.75, 0.75]],
    #                                     [[0.75, 0.50], [1.00, 0.75]],
    #                                     [[0.50, 0.75], [0.75, 1.00]],
    #                                     [[0.75, 0.75], [1.00, 1.00]],
    #                                     ])
    #
    #     torch.testing.assert_close(regions, expected_result)


    def test_place_signatures(self):

        regions = torch.Tensor([[[0.00, 0.00], [0.50, 0.50]],
                                [[0.50, 0.00], [1.00, 0.50]],
                                [[0.00, 0.50], [0.50, 1.00]],
                                [[0.50, 0.50], [1.00, 1.00]]
                                ])
        signatures = grid.place_signatures(regions)

        expected_result = torch.Tensor([[0.25, 0.25],
                                        [0.75, 0.25],
                                        [0.25, 0.75],
                                        [0.75, 0.75]
                                        ])

        torch.testing.assert_close(signatures, expected_result)


    def test_gmm_proba_mass_inside_hypercubes(self):

        m_1, m_2 = 0, 2
        w_1, w_2 = 0.3, 0.7

        means = torch.Tensor([[m_1], [m_2]])
        cov = torch.eye(1)
        weights = torch.Tensor([w_1, w_2])

        regions = torch.Tensor([[[-2.0], [0.0]],
                                [[0.0], [4.0]],
                                [[4.0], [10.0]]
                                ])

        probas = proba.compute_signature_probabilities(means, cov, weights, regions)

        gmm_1 = w_1 * ( norm.cdf(0.0, m_1, 1) - norm.cdf(-2.0, m_1, 1) ) + w_2 * ( norm.cdf(0.0, m_2, 1) - norm.cdf(-2.0, m_2, 1) )
        gmm_2 = w_1 * ( norm.cdf(4.0, m_1, 1) - norm.cdf(0.0, m_1, 1) ) + w_2 * ( norm.cdf(4.0, m_2, 1) - norm.cdf(0.0, m_2, 1) )
        gmm_3 = w_1 * (norm.cdf(10.0, m_1, 1) - norm.cdf(4.0, m_1, 1)) + w_2 * (norm.cdf(10.0, m_2, 1) - norm.cdf(4.0, m_2, 1))
        gmm_4 = 1 - gmm_1 - gmm_2 - gmm_3

        expected_result = torch.Tensor([gmm_1, gmm_2, gmm_3, gmm_4])

        torch.testing.assert_close(probas, expected_result)


    def test_envelopes_for_one_region(self):

        A = torch.Tensor(
            [[
                [0.5, 0.1],
                [0.1, 0.7]
            ]])

        f = LinearDynamics(A)

        regions = torch.Tensor([[[0.0, 0.0], [1.0, 1.0]]])
        print(regions.shape)

        envelopes = f.compute_hypercube_envelopes(regions)
        print(envelopes)
        print(envelopes.shape)

        expected_result = torch.Tensor([[[0.0, 0.0], [0.6, 0.8]]])

        torch.testing.assert_close(envelopes, expected_result)


    def test_envelopes_for_many_regions(self):

        A = torch.Tensor(
            [[
                [0.5, 0.1],
                [0.1, 0.7]
            ]])

        f = LinearDynamics(A)

        regions = torch.Tensor([[[0.0, 0.0], [1.0, 1.0]],
                                [[0.0, 1.0], [2.0, 3.0]]])
        print(regions.shape)

        envelopes = f.compute_hypercube_envelopes(regions)
        print(envelopes)
        print(envelopes.shape)

        expected_result = torch.Tensor([[[0.0, 0.0], [0.6, 0.8]],
                                        [[0.1, 0.7], [1.3, 2.3]]])

        torch.testing.assert_close(envelopes, expected_result)



