"""
noise_opt.py — conception du bruit par optimisation multi-objectif (verbe « optimiser » du sujet)

Problème : choisir (générateurs, zones, intensités, rondes) pour MAXIMISER la sécurité (log10 du nombre de
traces estimé contre l'attaquant le plus fort, sca_metrics.attack_strength) et MINIMISER le surcoût (%).
Chaque évaluation coûte une capture réelle (~3 min pour 4000 traces) : on ne peut pas tout essayer.

Méthode : ParEGO (Knowles, IEEE TEVC 2006) = optimisation bayésienne multi-objectif.
  * à chaque itération, un vecteur de poids lambda aléatoire transforme les 2 objectifs normalisés en un
    scalaire (Tchebycheff augmenté) ;
  * un processus gaussien (Rasmussen & Williams 2006 ; Snoek et al. NeurIPS 2012) est ajusté sur ce scalaire ;
  * on mesure la config candidate qui maximise l'amélioration espérée (Expected Improvement, Jones et al. 1998).
Le modèle GP est le « modèle de substitution » appris : il PRÉDIT la protection et le coût de configs jamais
mesurées. Résultat final : un front de Pareto MESURÉ, dont on extrait les niveaux de la politique embarquée.

Utilisation (banc) :
    import noise_opt as no
    space = no.candidate_space(masks=(0x0001, 0x0002, 0x0003, 0x0007, 0x07FF))
    ev = no.bench_evaluator(nt, key, n_traces=4000, outdir="traces/opt", baseline_cycles=95783)
    opt = no.ParEGO(space, ev, history_path="traces/opt/history.json")
    opt.initial_design(8)          # plan initial (aléatoire stratifié)
    opt.run(25)                     # 25 itérations ParEGO
    opt.report()                    # front de Pareto
    levels = no.select_levels(opt.history, k=3)
    no.save_levels(levels, "levels.json")
"""
import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np

# Générateurs autorisés par zone (copie de zone_eligible_mask du firmware)
ZONES = ("round_start", "subbytes_pre", "subbytes_byte", "mixcolumns", "addroundkey_pre", "addroundkey_byte", "round_end")
GENS = (None, "jitter", "dual_rail", "permutation", "alu")
ELIGIBLE = {0: (1,), 1: (3, 1), 2: (2, 1), 3: (4, 1, 3), 4: (1, 3), 5: (2, 1), 6: (1,)}   # firmware v8
PAIRS = tuple((z, g) for z in range(7) for g in ELIGIBLE[z])

# Grilles d'intensité brute (cycles ou répétitions selon le générateur) — d'après les Phases 1-2
# v8 : intensité de la permutation = nombre de colonnes factices de MixColumns (0 à 4 ; x4 octets factices
# dans SubBytes). 0 = mélange sans créneau factice.
RAW_GRID = {1: (10, 25, 50, 100, 200, 400, 800), 2: (3, 6, 12, 25, 50, 100), 3: (0, 1, 2, 4), 4: (5, 20, 50, 200)}
DEFAULT_MASKS = (0x0001, 0x0002, 0x0003, 0x0006, 0x0007, 0x07FF)


@dataclass(frozen=True)
class Config:
    comps: tuple                 # ((zone, gen), ...) triés
    raw: tuple                   # ((gen, cycles), ...) triés, un par générateur utilisé
    mask: int = 0x07FF

    @property
    def name(self):
        r = dict(self.raw)
        parts = ["%s@%s:%d" % (GENS[g], ZONES[z], r[g]) for z, g in self.comps]
        return "+".join(parts) + "|m%03X" % self.mask

    def to_dict(self):
        return dict(comps=[list(c) for c in self.comps], raw=[list(x) for x in self.raw], mask=self.mask, name=self.name)

    @staticmethod
    def from_dict(d):
        return Config(tuple(tuple(c) for c in d["comps"]), tuple(tuple(x) for x in d["raw"]), int(d["mask"]))

    def setup(self):
        import sca_capture as sc
        return sc.make_setup(forces=[(z, g, 0) for z, g in self.comps], raw=dict(self.raw), round_mask=self.mask)


def candidate_space(max_comps=2, masks=DEFAULT_MASKS, pairs=PAIRS, raw_grid=RAW_GRID):
    """Énumère les configs : 1 ou 2 composantes (zone, générateur) ; l'intensité brute est par générateur
    (contrainte du firmware : g_raw_cycles[g] est partagé entre zones)."""
    out = []
    for k in range(1, max_comps + 1):
        for comps in itertools.combinations(pairs, k):
            gens = sorted({g for _, g in comps})
            for raws in itertools.product(*[raw_grid[g] for g in gens]):
                for m in masks:
                    out.append(Config(tuple(sorted(comps)), tuple(zip(gens, raws)), int(m)))
    return out


def encode(cfg):
    """Vecteur de description d'une config pour le GP : présence des paires (zone, gén.), log(1+intensité)
    par générateur, bits du masque de rondes."""
    v = np.zeros(len(PAIRS) + 4 + 11)
    for c in cfg.comps:
        v[PAIRS.index(tuple(c))] = 1.0
    for g, cy in cfg.raw:
        v[len(PAIRS) + g - 1] = np.log1p(cy) / np.log1p(800)
    for r in range(11):
        v[len(PAIRS) + 4 + r] = (cfg.mask >> r) & 1
    return v


# --------------------------------------------------------------------------
# Évaluateur sur banc
# --------------------------------------------------------------------------
def bench_evaluator(nt, key, n_traces=4000, outdir="traces/opt", baseline_cycles=None, seed=42,
                    window=(3000, 11000), bytes_=(0, 5, 10, 15), widths=(1, 4, 16, 64, 256), alpha=0.9999,
                    clean=None, max_shift=3000):
    """Retourne f(cfg) -> dict(log10_n, overhead_pct, ...). Capture réelle + sca_metrics.evaluate_rho_campaign.
    baseline_cycles : moyenne de trig_count SANS bruit, mesurée dans la MÊME session (même build)."""
    import sca_capture as sc
    import sca_metrics as sm
    if clean is None:
        print("ATTENTION : sans `clean` (sm.clean_reference), seuls A1/A2 sont utilisés -> la gigue paraît forte à tort")
    if baseline_cycles is None:
        raise ValueError("mesurez baseline_cycles dans cette session (campagne sans bruit, 500 traces)")

    def _f(cfg):
        safe = cfg.name.replace("@", "_").replace(":", "").replace("+", "__").replace("|", "_")
        path = sc.capture_campaign(nt, safe, n_traces, key, seed=seed, outdir=outdir, setup=cfg.setup(),
                                   meta=dict(config=cfg.to_dict()))
        r = sm.evaluate_rho_campaign(path, bytes_=bytes_, window=window, widths=widths, alpha=alpha,
                                     baseline_cycles=baseline_cycles, clean=clean, max_shift=max_shift)
        return dict(log10_n=r["log10_n_est"], overhead_pct=r["overhead_pct"], rho_star=r["rho_star"],
                    resolution=r["resolution"], censored=r["censored"], worst_byte=r["worst_byte"],
                    attacker=r["per_byte"][r["worst_byte"]]["attacker"], path=path)
    return _f


# --------------------------------------------------------------------------
# ParEGO
# --------------------------------------------------------------------------
def _expected_improvement(mu, sd, best, xi=0.01):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-9)
    imp = best - mu - xi                      # minimisation
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


class ParEGO:
    def __init__(self, space, evaluate, history_path=None, seed=0, rho_aug=0.05, s=10, max_overhead=None):
        """max_overhead (%) : contrainte de coût. Un 2e GP apprend log(1+surcoût) ; les candidats dont le
        surcoût PRÉDIT dépasse la limite ne sont pas proposés (évite de gaspiller des captures)."""
        self.max_overhead = max_overhead
        self.space = list(space)
        self.X = np.stack([encode(c) for c in self.space])
        self.evaluate = evaluate
        self.history_path = history_path
        self.rng = np.random.default_rng(seed)
        self.rho_aug = rho_aug
        self.lambdas = np.array([[i / s, 1 - i / s] for i in range(s + 1)])   # Knowles 2006, 2 objectifs
        self.history = []
        if history_path and os.path.exists(history_path):
            with open(history_path) as f:
                self.history = json.load(f)
            print("historique repris : %d évaluations" % len(self.history))

    # -- gestion de l'historique
    def _done(self):
        return {h["config"]["name"] for h in self.history}

    def _record(self, cfg, res, how):
        h = dict(config=cfg.to_dict(), how=how, **res)
        self.history.append(h)
        if self.history_path:
            os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=1, default=float)
        print("[%2d] %-55s log10N=%5.2f%s  surcoût=%6.1f%%  (%s)" % (
            len(self.history), cfg.name, res["log10_n"], "+" if res.get("censored") else " ",
            res["overhead_pct"], how))

    def _objectives(self):
        F = np.array([[-h["log10_n"], h["overhead_pct"]] for h in self.history], float)
        lo, hi = F.min(0), F.max(0)
        return (F - lo) / np.where(hi > lo, hi - lo, 1.0)

    # -- étapes
    def initial_design(self, n=8):
        """Plan initial : une config par paire (zone, générateur) tirée au hasard, pour couvrir l'espace."""
        done = self._done()
        by_pair = {}
        for i, c in enumerate(self.space):
            if len(c.comps) == 1:
                by_pair.setdefault(c.comps[0], []).append(i)
        keys = list(by_pair)
        self.rng.shuffle(keys)
        k = 0
        for p in keys:
            if k >= n:
                break
            i = int(self.rng.choice(by_pair[p]))
            if self.space[i].name in done:
                continue
            self._record(self.space[i], self.evaluate(self.space[i]), "initial")
            k += 1

    def suggest(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        F = self._objectives()
        lam = self.lambdas[self.rng.integers(len(self.lambdas))]
        g = np.max(F * lam, axis=1) + self.rho_aug * (F @ lam)          # Tchebycheff augmenté
        name_to_i = {c.name: i for i, c in enumerate(self.space)}
        Xh = np.stack([encode(Config.from_dict(h["config"])) for h in self.history])
        kern = ConstantKernel(1.0) * Matern(length_scale=np.ones(Xh.shape[1]), nu=2.5) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(self.rng.integers(1 << 31)))
        gp.fit(Xh, g)
        mask = np.ones(len(self.space), bool)
        for h in self.history:
            i = name_to_i.get(h["config"]["name"])
            if i is not None:
                mask[i] = False
        if self.max_overhead is not None and len(self.history) >= 4:
            yc = np.log1p(np.maximum([h["overhead_pct"] for h in self.history], 0.0))
            gpc = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=np.ones(Xh.shape[1]), nu=2.5)
                                           + WhiteKernel(1e-2), normalize_y=True, random_state=0).fit(Xh, yc)
            mask[mask] = np.expm1(gpc.predict(self.X[mask])) <= self.max_overhead
            if not mask.any():
                raise RuntimeError("aucun candidat sous max_overhead")
        mu, sd = gp.predict(self.X[mask], return_std=True)
        ei = _expected_improvement(mu, sd, g.min())
        cand = np.flatnonzero(mask)[int(np.argmax(ei))]
        return self.space[cand], lam, gp

    def run(self, n_iter=20):
        for _ in range(n_iter):
            cfg, lam, _ = self.suggest()
            self._record(cfg, self.evaluate(cfg), "parego l=(%.1f,%.1f)" % tuple(lam))

    def pareto(self):
        import sca_metrics as sm
        c = [h["overhead_pct"] for h in self.history]
        s = [h["log10_n"] for h in self.history]
        return [self.history[i] for i in sm.pareto_front(c, s)]

    def report(self):
        P = self.pareto()
        print("\nFront de Pareto mesuré (%d points sur %d évaluations) :" % (len(P), len(self.history)))
        print("%-55s %9s %9s %s" % ("config", "surcoût", "log10 N", "attaquant"))
        for h in P:
            print("%-55s %+8.1f%% %8.2f%s %s" % (h["config"]["name"], h["overhead_pct"], h["log10_n"],
                                                 "+" if h.get("censored") else " ", h.get("attacker", "")))
        return P


# --------------------------------------------------------------------------
# Niveaux de la politique embarquée
# --------------------------------------------------------------------------
def select_levels(history, k=3, min_gain=0.3, max_overhead=None):
    """Choisit k niveaux sur le front de Pareto, espacés en log10 N (au moins `min_gain` décade d'écart).
    leak : atanh(rho)² de l'attaquant le plus fort ; si la mesure est censurée, on prend rho = résolution
    (majorant : le budget de fuite est alors CONSERVATEUR)."""
    import sca_metrics as sm
    if max_overhead is not None:
        history = [h for h in history if h["overhead_pct"] <= max_overhead]
    c = [h["overhead_pct"] for h in history]
    s = [h["log10_n"] for h in history]
    P = [history[i] for i in sm.pareto_front(c, s)]
    if not P:
        return []
    targets = np.linspace(P[0]["log10_n"], P[-1]["log10_n"], k)
    chosen, last = [], -np.inf
    for t in targets:
        h = min(P, key=lambda q: abs(q["log10_n"] - t))
        if h in chosen or h["log10_n"] < last + min_gain and chosen:
            continue
        chosen.append(h); last = h["log10_n"]
    levels = []
    for h in chosen:
        rho = max(h["rho_star"], h["resolution"]) if h.get("censored") else h["rho_star"]
        levels.append(dict(config=h["config"], overhead_pct=float(h["overhead_pct"]), log10_n=float(h["log10_n"]),
                           rho=float(rho), leak=float(sm.leak_per_trace(rho)), leak_q16=sm.leak_q16(rho),
                           censored=bool(h.get("censored", False))))
    levels.sort(key=lambda L: L["overhead_pct"])
    return levels


def save_levels(levels, path):
    with open(path, "w") as f:
        json.dump(levels, f, indent=1)


def load_levels(path):
    with open(path) as f:
        return json.load(f)
