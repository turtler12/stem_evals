"""
Question & Answer loading pipeline.
Loads questions from extracted JSON (data/extracted_questions.json) or falls back
to hardcoded 8.033 Final Practice Problems for backward compatibility.
"""

import json
import os
from pathlib import Path

EXTRACTED_QUESTIONS_PATH = Path(__file__).parent.parent / "data" / "extracted_questions.json"

PHYSICS_FORMULA_SHEET = r"""
Formula sheet (provided to students):
Lorentz transformations: Primed frame moves with v = ve_x relative to unprimed frame.
[c\Delta t', \Delta x', \Delta y', \Delta z']^T = \Lambda [c\Delta t, \Delta x, \Delta y, \Delta z]^T
where \Lambda = [[gamma, -beta*gamma, 0, 0], [-beta*gamma, gamma, 0, 0], [0,0,1,0], [0,0,0,1]]

Important 4-vectors:
4-velocity: U^mu = dx^mu / d\tau
4-momentum: P^a = m U^a, P^a P_a = -E^2 + |p|^2 = -m^2
4-force: F^a = dP^a / d\tau
4-acceleration: A^a = dU^a / d\tau

Addition of velocities:
(u)^x = ((u')^x + v) / (1 + (u')^x v)
(u)^y = (u')^y / (gamma(1 + (u')^x v))
(u)^z = (u')^z / (gamma(1 + (u')^x v))

Faraday tensor: F^{alpha beta} = [[0, E^x, E^y, E^z],[-E^x,0,B^z,-B^y],[-E^y,-B^z,0,B^x],[-E^z,B^y,-B^x,0]]
Electromagnetic force law: dP^a/d\tau = q F^{ab} U_b

Transformation of fields:
E'_|| = E_||, E'_perp = gamma(E_perp + v x B_perp)
B'_|| = B_||, B'_perp = gamma(B_perp - v x E_perp / c^2)

Uniformly Accelerated Observers:
t(tau) = (1/alpha) sinh(alpha * tau)
x(tau) = (1/alpha) cosh(alpha * tau)

Doppler Shift: f_O = gamma * f_S * (1 - v cos(theta_S))

Geodesic equation: U^a nabla_a U^b = 0, where nabla_a U^b = partial_a U^b + Gamma^b_{ac} U^c

Curvature tensors: R_{bd} = R^a_{bad}, R = R^a_a, G_{ab} = R_{ab} - (1/2) R g_{ab}, K = R_{abcd} R^{abcd}
Einstein equation: G_{ab} + Lambda g_{ab} = 8 pi G_N T_{ab}

Schwarzschild metric:
ds^2 = -(1 - 2GM/r) dt^2 + dr^2/(1 - 2GM/r) + r^2(d\theta^2 + sin^2\theta d\phi^2)
Event horizon at r = 2GM.
"""

QUESTIONS = [
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1a",
        "question_text": "Short answer: Describe one way in which Newtonian gravity fails to predict empirical results.",
        "solution_text": "Precession of Mercury, gravitational lensing, clocks at different heights running at different rates, etc.",
        "topic_tags": ["general_relativity", "short_answer", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1b",
        "question_text": "Short answer: Define a null vector.",
        "solution_text": "A vector whose norm satisfies K^a K_a = 0.",
        "topic_tags": ["special_relativity", "short_answer", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1c",
        "question_text": "Short answer: What is the Ricci scalar of Minkowski space in any coordinates?",
        "solution_text": "Since R^alpha_{beta gamma delta} = 0 in Cartesian coordinates of R^{3,1}, R = 0 in Cartesian coordinates. But R is coordinate invariant, so R = 0 for R^{3,1} in all coordinates.",
        "topic_tags": ["general_relativity", "short_answer", "curvature"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1d",
        "question_text": "Short answer: Why are global inertial frames not sufficient to explain gravity?",
        "solution_text": "Global inertial frames do not exist in gravity: even free falling frames accelerate.",
        "topic_tags": ["general_relativity", "short_answer", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1e",
        "question_text": r"Short answer: Why is \partial_a not a vector in special relativity?",
        "solution_text": r"\partial_a does not transform like a vector under Lorentz transformations.",
        "topic_tags": ["special_relativity", "short_answer", "tensors"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-1f",
        "question_text": r"Short answer: Why is \partial_a not a 1-form in general relativity?",
        "solution_text": r"Since \partial_mu g_{nu rho}|_p = 0 for locally inertial coordinates at p, this would mean \partial_mu g_{nu rho} = 0 in any coordinate system at all p, so \partial_mu g_{nu rho} can't be a tensor.",
        "topic_tags": ["general_relativity", "short_answer", "tensors"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-2a",
        "question_text": r"""Consider the metric:
ds^2 = -(1 + r^2/l^2) dt^2 + dr^2/(1 + r^2/l^2) + r^2 d\Omega^2

Note that this is not Schwarzschild. The Riemann tensor of this metric is:
R_{abcd} = -(1/l^2)(g_{ac} g_{bd} - g_{ad} g_{bc})

(a) Compute the Ricci tensor of this spacetime.""",
        "solution_text": r"R_{bd} = R_{abcd} g^{ac} = -(1/l^2)(g_{ac} g^{ac} g_{bd} - g_{ad} g_{bc} g^{ac}) = -(1/l^2)(4 g_{bd} - delta^c_d g_{bc}) = -(1/l^2)(4 g_{bd} - g_{bd}) = -(3/l^2) g_{bd}",
        "topic_tags": ["general_relativity", "curvature", "tensor_computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-2b",
        "question_text": r"""Consider the metric:
ds^2 = -(1 + r^2/l^2) dt^2 + dr^2/(1 + r^2/l^2) + r^2 d\Omega^2

The Riemann tensor is R_{abcd} = -(1/l^2)(g_{ac} g_{bd} - g_{ad} g_{bc}).

(b) Compute the Ricci scalar of this spacetime.""",
        "solution_text": r"R = R_{bd} g^{bd} = -(3/l^2) g_{bd} g^{bd} = -12/l^2",
        "topic_tags": ["general_relativity", "curvature", "tensor_computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-2c",
        "question_text": r"""Consider the metric:
ds^2 = -(1 + r^2/l^2) dt^2 + dr^2/(1 + r^2/l^2) + r^2 d\Omega^2

The Riemann tensor is R_{abcd} = -(1/l^2)(g_{ac} g_{bd} - g_{ad} g_{bc}), the Ricci tensor is R_{ab} = -(3/l^2) g_{ab}, and the Ricci scalar is R = -12/l^2.

(c) Compute the Einstein tensor of this spacetime.""",
        "solution_text": r"G_{ab} = R_{ab} - (1/2) R g_{ab} = -(3/l^2) g_{ab} + (6/l^2) g_{ab} = (3/l^2) g_{ab}",
        "topic_tags": ["general_relativity", "curvature", "tensor_computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-2d",
        "question_text": r"""Consider the metric:
ds^2 = -(1 + r^2/l^2) dt^2 + dr^2/(1 + r^2/l^2) + r^2 d\Omega^2

The Riemann tensor is R_{abcd} = -(1/l^2)(g_{ac} g_{bd} - g_{ad} g_{bc}).

(d) Compute the Kretschmann scalar of this spacetime.""",
        "solution_text": r"K = R_{abcd} R^{abcd} = (1/l^4)(g_{ac}g_{bd} - g_{ad}g_{bc})(g^{ac}g^{bd} - g^{ad}g^{bc}) = (1/l^4)(32 - delta^c_d delta^d_c - delta^d_c delta^c_d) = 24/l^4",
        "topic_tags": ["general_relativity", "curvature", "tensor_computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-3a",
        "question_text": r"""Consider the following curve in the Schwarzschild geometry for r > R_S: r = r_0, theta = theta_0, phi = phi_0.

(a) Write an expression for a tangent 4-vector to this curve.""",
        "solution_text": r"The curve is (t, r, theta, phi) = (t, r_0, theta_0, phi_0). A particular tangent is U^mu = (1, 0, 0, 0).",
        "topic_tags": ["general_relativity", "schwarzschild", "geodesics"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-3b",
        "question_text": r"""Consider the following curve in the Schwarzschild geometry for r > R_S: r = r_0, theta = theta_0, phi = phi_0, with tangent 4-vector U^mu = (1, 0, 0, 0).

(b) Compute the norm of this tangent 4-vector. Is it timelike, spacelike, or null?""",
        "solution_text": r"U^mu U^nu g_{mu nu} = -(1 - 2GM/r). For r > R_s, this is negative, so U^mu is timelike.",
        "topic_tags": ["general_relativity", "schwarzschild", "geodesics"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-3c",
        "question_text": r"""Consider the following curve in the Schwarzschild geometry for r > R_S: r = r_0, theta = theta_0, phi = phi_0, with tangent 4-vector U^mu = (1, 0, 0, 0).

(c) Recall that the 4-velocity of a particle always satisfies U^a U_a = -1. Is this true of the tangent vector you wrote down above? If not, how do you explain the discrepancy?""",
        "solution_text": r"It is not true. U^mu is not a 4-velocity because we did not parametrize it by proper time tau: we parametrized it by coordinate time t.",
        "topic_tags": ["general_relativity", "schwarzschild", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-4",
        "question_text": "How much older is your head than your feet?",
        "solution_text": r"The time delay between clocks at different heights (approximating g = constant): Delta_tau_l / Delta_tau_0 = e^{gl/c^2}. If you are 20 years old according to your feet and have a height of 1.75m, then Delta_tau_l - Delta_tau_0 = Delta_tau_0 (e^{gl/c^2} - 1) = 1.4 * 10^{-7} seconds.",
        "topic_tags": ["general_relativity", "gravitational_time_dilation", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-5",
        "question_text": "A police car traveling at 0.5c chases bank robbers traveling at 0.75c. Realizing they can't catch up, the police decide to shoot out the robbers' tires. What is the minimum speed required for their bullets to leave the officers' guns in order for the robbers to not outrun the bullets as well?",
        "solution_text": r"In the police car frame: v'_c = (v_p - v_c)/(1 - v_p * v_c) = (0.75 - 0.5)/(1 - 0.75*0.5) = 0.25/0.625 = 2/5. The bullets need to be traveling at v'_bullet > (2/5)c in the police frame.",
        "topic_tags": ["special_relativity", "velocity_addition", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-6",
        "question_text": r"A beam of muons, all of which are known to have the same energy, is observed to have the property that on average each muon travels 2000 meters before decaying. A student who hasn't taken 8.033 concludes that the muon has a mean lifetime of 2 * 10^{-6} seconds and concludes that the muons traveled at 10^9 meters/second. Identify the student's error, and compute the correct muon velocity.",
        "solution_text": r"The student neglected to account for time dilation: t = gamma * tau_0, where tau_0 = 2 * 10^{-6} s (rest lifetime). v * t = v * gamma * tau_0 = 2000 m. Using the definition of gamma: v^2 / (1 - v^2/c^2) = (2000m / (2*10^{-6}s))^2. We find v = 2.87 * 10^8 m/s.",
        "topic_tags": ["special_relativity", "time_dilation", "muon_decay", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-7a",
        "question_text": r"""Consider a uniformly accelerated observer with proper acceleration alpha.

(a) Compute the 4-velocity components U^mu of the observer in Cartesian Minkowski coordinates (you may leave it in terms of tau).""",
        "solution_text": r"The trajectory is (t, x, y, z) = ((1/alpha) sinh(alpha*tau), (1/alpha) cosh(alpha*tau), 0, 0), so the 4-velocity is U^mu = dx^mu/d_tau = (cosh(alpha*tau), sinh(alpha*tau), 0, 0).",
        "topic_tags": ["special_relativity", "acceleration", "4-velocity", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-7b",
        "question_text": r"""Consider a uniformly accelerated observer with proper acceleration alpha, with 4-velocity U^mu = (cosh(alpha*tau), sinh(alpha*tau), 0, 0).

(b) Using the fact that in Cartesian coordinates of Minkowski space, all Christoffel symbols vanish, show using the geodesic equation that uniformly accelerated observers do not follow geodesics.""",
        "solution_text": r"U^mu nabla_mu U^nu = U^mu (partial_mu U^nu + Gamma^nu_{mu rho} U^rho) = U^mu partial_mu U^nu = (dx^mu/d_tau) partial_mu U^nu = dU^nu/d_tau != 0. Since dU^nu/d_tau = (alpha * sinh(alpha*tau), alpha * cosh(alpha*tau), 0, 0) != 0, the trajectories do not solve the geodesic equation.",
        "topic_tags": ["general_relativity", "geodesics", "acceleration", "proof"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-7c",
        "question_text": r"""Recall that U^a nabla_a U^b is the directional derivative of the 4-velocity in the direction of the 4-velocity.

(c) Write down a general expression for the directional derivative of an arbitrary vector V^a in the direction of an arbitrary vector W^a.""",
        "solution_text": r"W^a nabla_a V^b",
        "topic_tags": ["general_relativity", "covariant_derivative", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8a",
        "question_text": r"""Kruskal Coordinates. Introduce the Regge-Wheeler "tortoise coordinate":
r_* = integral of r dr / (r - 2GM)

(a) Write out r_* as a function of r. What happens to r_* as r approaches 2GM? Why is this called a "tortoise" coordinate?""",
        "solution_text": r"r_* = r + 2GM ln(r/(2GM) - 1). r_* goes to -infinity as r -> 2GM. This is a very slow coordinate (like a tortoise approaching the horizon).",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8b",
        "question_text": r"""Define lightcone coordinates: u = t - r_*, v = t + r_*. These are null coordinates.

(b) Which one is constant along light rays moving towards larger r and which one is constant along light rays moving towards smaller r?""",
        "solution_text": "u = constant along lines moving towards larger r. v = constant along lines moving towards smaller r.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "conceptual"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8c",
        "question_text": r"""Using lightcone coordinates u = t - r_*, v = t + r_*:

(c) Rewrite the line element in the t-r directions in terms of du, dv, and r(u,v), without evaluating the latter. What are g_{uu} and g_{vv}?""",
        "solution_text": r"ds^2 = -(1 - 2GM/r(u,v)) du dv + r(u,v)^2 d\Omega^2, and g_{uu} = 0 = g_{vv}.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8d",
        "question_text": r"""(d) Express (1 - 2GM/r) in terms of a function of r that doesn't vanish at r = R_S and the function e^{(v-u)/(4GM)}. Plug this into the line element in the t-r directions to replace (1 - 2GM/r).""",
        "solution_text": r"From r_* definition: r + 2GM ln(r/(2GM) - 1) = (v-u)/2, so e^{r/(2M)} (r/(2M) - 1) = e^{(v-u)/(4M)}. Thus 1 - 2M/r = (2M/r) * ... and ds^2 = -(2M/r) e^{-r/(2M)} e^{(v-u)/(4M)} du dv.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8e",
        "question_text": r"""Define U = -e^{-u/(4GM)}, V = e^{v/(4GM)}.

(e) Rewrite the t-r line element in terms of U and V, keeping the r^2 d\Omega^2 term intact.""",
        "solution_text": r"dU = (1/(4M)) e^{-u/(4M)} du, dV = (1/(4M)) e^{v/(4M)} dv. ds^2 = -(32(GM)^3/r) e^{-r/(2M)} dU dV + r^2 d\Omega^2.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8f",
        "question_text": r"""These Kruskal coordinates are regular everywhere except at r = 0.

(f) The range of U and V is constrained only by r(U,V) > 0. What is the range of U and V so that r(U,V) > 0?""",
        "solution_text": r"U in (-infinity, infinity) and V in (-infinity, infinity), but UV < 1.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8g",
        "question_text": r"""(g) Show that r(U,V) = 0 corresponds to two different singularities. One is the singularity in the future of an observer in region II. The other is a singularity in the past of an observer in region IV.""",
        "solution_text": r"e^{r/(2M)} (r/(2M) - 1) = e^{(v-u)/(2M)} = -VU. When r = 0, UV = 1. r = 0 occurs twice. U > 0, V > 0 and U < 0, V < 0 correspond to two singularities.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal", "proof"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-8h",
        "question_text": "(h) Find the two asymptotic regions (the two ends of the wormhole) in terms of U and V.",
        "solution_text": r"When r -> +infinity, UV -> -infinity. The left asymptotic region is U -> -infinity, and the right asymptotic region is V -> -infinity.",
        "topic_tags": ["general_relativity", "schwarzschild", "kruskal"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-9a",
        "question_text": r"""Higgs decay: The rest mass of the Higgs is about m_h c^2 = 125 GeV. The most common Higgs decay process is into two photons: h -> gamma gamma.

(a) In the rest frame of the Higgs, compute the energy of each photon after the decay process.""",
        "solution_text": r"In the rest frame of the Higgs, the photons have equal and opposite 3-momentum. The total 4-momentum is p^mu = (m_H, 0, 0, 0). This is conserved, so p^0 = 2 p^0_gamma = m_H, so each photon has energy p^0_gamma = m_H/2 = 62.5 GeV.",
        "topic_tags": ["special_relativity", "particle_physics", "4-momentum", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-9b",
        "question_text": r"""Higgs decay h -> gamma gamma. m_h c^2 = 125 GeV.

(b) Suppose in the lab frame the Higgs has energy E_h and one photon's energy is measured to be E_gamma. What is the angle that the photon makes with the original path of the Higgs?""",
        "solution_text": r"Using 4-momentum conservation, p^a_{gamma'} = p^a_H - p^a_gamma, and p^a_{gamma'} p_{gamma' a} = 0. This gives 0 = -m_H^2 + 2 E_H E_gamma - 2 E_gamma cos(theta) sqrt(-m_H^2 + E_H^2). Solving: theta = cos^{-1}[(2 E_H E_gamma - m_H^2) / (2 E_gamma sqrt(-m_H^2 + E_H^2))].",
        "topic_tags": ["special_relativity", "particle_physics", "4-momentum", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-10a",
        "question_text": r"""Comparing Metrics in Space. Given the line element (purely spatial):
ds^2 = du^2 + cosh^2(u) dv^2

And the flat space metric: ds^2 = dr^2 + r^2 d\phi^2

(a) What are the circumferences of a circle in both spaces? (A circle = set of all points at given distance from a center. Use u as radius in the first metric; identify v from 0 to 2pi.)""",
        "solution_text": r"Euclidean: L_E = integral from 0 to 2pi of sqrt(dr^2 + r^2 d\theta^2) = r * 2pi = 2*pi*r. Lobachevsky: L_L = integral from 0 to 2pi of sqrt(du^2 + cosh^2(u) dv^2) = integral of cosh(u) dv = 2*pi*cosh(u).",
        "topic_tags": ["general_relativity", "differential_geometry", "metric", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-10b",
        "question_text": r"""Comparing Metrics in Space.
ds^2 = du^2 + cosh^2(u) dv^2 (Lobachevsky) vs ds^2 = dr^2 + r^2 d\phi^2 (Euclidean).

(b) What are the areas of a circle in both spaces?""",
        "solution_text": r"Euclidean: A_E = integral from 0 to 2pi, 0 to R of r dr d\theta = pi R^2. Lobachevsky: A_L = integral from 0 to 2pi, 0 to U of cosh(u) du dv = 2*pi*sinh(U).",
        "topic_tags": ["general_relativity", "differential_geometry", "metric", "computation"],
    },
    {
        "course": "8.033",
        "source_file": "Final Practice Problems.pdf",
        "question_id": "8.033-final-practice-10c",
        "question_text": r"""Comparing Metrics in Space.
ds^2 = du^2 + cosh^2(u) dv^2 (Lobachevsky) vs ds^2 = dr^2 + r^2 d\phi^2 (Euclidean).

(c) How do you explain the differences?""",
        "solution_text": "Euclidean space is flat while Lobachevsky space is not. Lobachevsky space is hyperbolic (kind of like a wormhole). Circumference and area increase much faster with radius!",
        "topic_tags": ["general_relativity", "differential_geometry", "conceptual"],
    },
]


def load_all_questions() -> list[dict]:
    """Load questions from extracted JSON if available, else fall back to hardcoded."""
    if EXTRACTED_QUESTIONS_PATH.exists():
        with open(EXTRACTED_QUESTIONS_PATH) as f:
            return json.load(f)
    return QUESTIONS


def get_questions(question_ids: list[str] | None = None,
                  course: str | None = None) -> list[dict]:
    """Return questions, optionally filtered by IDs and/or course."""
    all_qs = load_all_questions()
    if course:
        all_qs = [q for q in all_qs if q["course"] == course]
    if question_ids is not None:
        all_qs = [q for q in all_qs if q["question_id"] in question_ids]
    return all_qs


def save_questions_json(output_path: str = "results/questions.json"):
    """Save all questions to a JSON file."""
    all_qs = load_all_questions()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qs, f, indent=2)
    return output_path


def get_formula_sheet(course: str = "8.033") -> str | None:
    """Return formula sheet for the given course, or None for non-physics courses."""
    if course == "8.033":
        return PHYSICS_FORMULA_SHEET
    return None


if __name__ == "__main__":
    path = save_questions_json()
    qs = load_all_questions()
    print(f"Saved {len(qs)} questions to {path}")
    courses = set(q["course"] for q in qs)
    for c in sorted(courses):
        count = sum(1 for q in qs if q["course"] == c)
        print(f"  {c}: {count} questions")
