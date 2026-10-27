"""
Étape 1 — Exploration de CartPole-v1 avec une politique aléatoire.

Objectifs :
  1. Inspecter les espaces d'observation et d'action.
  2. Faire tourner une politique uniformément aléatoire sur N épisodes.
  3. Mesurer la récompense par épisode (= sa durée, +1 par pas) : c'est la
     ligne de base que DQN devra battre (cible : moyenne > 475 sur 100 épisodes).
  4. Observer la distribution des 4 variables d'état et la cause de fin
     d'épisode (angle, position ou troncature à 500 pas).

Usage :
    python etape1_exploration_aleatoire.py --episodes 1000 --seed 0
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOMS_ETAT = ["Position chariot (x)", "Vitesse chariot (ẋ)",
             "Angle pendule (θ, rad)", "Vitesse angulaire (θ̇)"]

# Seuils de terminaison de CartPole-v1 (cf. code source de Gymnasium)
SEUIL_X = 2.4
SEUIL_THETA = 12 * 2 * np.pi / 360  # 12° ≈ 0.2095 rad


def decrire_environnement(env):
    print("=" * 60)
    print("ENVIRONNEMENT :", env.spec.id)
    print("=" * 60)
    print("Espace d'observation :", env.observation_space)
    print("  bornes basses :", env.observation_space.low)
    print("  bornes hautes :", env.observation_space.high)
    print("Espace d'action      :", env.action_space, "(0 = gauche, 1 = droite)")
    print("Pas max par épisode  :", env.spec.max_episode_steps)
    print("Seuil 'résolu'       :", env.spec.reward_threshold)
    print()


def cause_de_fin(obs, terminated, truncated):
    if truncated and not terminated:
        return "troncature (500 pas)"
    if abs(obs[0]) > SEUIL_X:
        return "chariot hors piste (|x| > 2.4)"
    if abs(obs[2]) > SEUIL_THETA:
        return "pendule tombé (|θ| > 12°)"
    return "autre"


def executer(n_episodes, seed):
    env = gym.make("CartPole-v1")
    decrire_environnement(env)
    env.action_space.seed(seed)

    retours, toutes_obs, causes = [], [], {}
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        retour, fini = 0.0, False
        while not fini:
            toutes_obs.append(obs)
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            retour += r
            fini = terminated or truncated
        c = cause_de_fin(obs, terminated, truncated)
        causes[c] = causes.get(c, 0) + 1
        retours.append(retour)
    env.close()
    return np.array(retours), np.array(toutes_obs), causes


def resumer(retours, obs, causes, n_episodes):
    moy100 = np.convolve(retours, np.ones(100) / 100, mode="valid")
    resume = {
        "episodes": n_episodes,
        "transitions_totales": int(len(obs)),
        "retour_moyen": float(retours.mean()),
        "retour_ecart_type": float(retours.std()),
        "retour_median": float(np.median(retours)),
        "retour_min": float(retours.min()),
        "retour_max": float(retours.max()),
        "meilleure_moyenne_glissante_100": float(moy100.max()),
        "causes_de_fin": causes,
        "etat_min": obs.min(axis=0).round(4).tolist(),
        "etat_max": obs.max(axis=0).round(4).tolist(),
        "etat_moyenne": obs.mean(axis=0).round(4).tolist(),
        "etat_ecart_type": obs.std(axis=0).round(4).tolist(),
    }
    print("RÉSULTATS — politique aléatoire")
    print("-" * 60)
    print(f"Épisodes             : {n_episodes}  ({len(obs)} transitions)")
    print(f"Retour moyen         : {retours.mean():.2f} ± {retours.std():.2f}")
    print(f"Médiane / min / max  : {np.median(retours):.0f} / "
          f"{retours.min():.0f} / {retours.max():.0f}")
    print(f"Meilleure moy. 100 ép: {moy100.max():.2f}   (objectif DQN : > 475)")
    print("Causes de fin        :")
    for c, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"   {c:<32} {n:>5}  ({100 * n / n_episodes:.1f} %)")
    print("Statistiques des variables d'état (min / max / moyenne / écart-type) :")
    for i, nom in enumerate(NOMS_ETAT):
        print(f"   {nom:<26} {obs[:, i].min():+.3f} / {obs[:, i].max():+.3f} / "
              f"{obs[:, i].mean():+.3f} / {obs[:, i].std():.3f}")
    return resume


def tracer(retours, obs, chemin):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("CartPole-v1 — politique aléatoire", fontsize=14)

    ax = axes[0, 0]
    ax.plot(retours, lw=0.6, alpha=0.5, color="#4C72B0", label="par épisode")
    if len(retours) >= 100:
        m = np.convolve(retours, np.ones(100) / 100, mode="valid")
        ax.plot(np.arange(99, len(retours)), m, color="#C44E52", lw=2,
                label="moyenne glissante 100")
    ax.axhline(475, ls="--", color="gray", label="seuil 'résolu' (475)")
    ax.set(title="Retour par épisode", xlabel="épisode", ylabel="retour")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(retours, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(retours.mean(), color="#C44E52", lw=2,
               label=f"moyenne = {retours.mean():.1f}")
    ax.set(title="Distribution des retours", xlabel="retour", ylabel="épisodes")
    ax.legend(fontsize=8)

    seuils = [SEUIL_X, None, SEUIL_THETA, None]
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for i, (r, c) in enumerate(positions):
        ax = axes[r, c]
        ax.hist(obs[:, i], bins=60, color="#55A868", edgecolor="white")
        if seuils[i] is not None:
            for s in (-seuils[i], seuils[i]):
                ax.axvline(s, ls="--", color="#C44E52")
        ax.set(title=NOMS_ETAT[i], ylabel="fréquence")

    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    print(f"\nFigure enregistrée : {chemin}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="resultats/etape1")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    retours, obs, causes = executer(args.episodes, args.seed)
    resume = resumer(retours, obs, causes, args.episodes)
    resume["seed"] = args.seed
    tracer(retours, obs, out / "politique_aleatoire.png")
    with open(out / "resume.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    print(f"Résumé enregistré  : {out / 'resume.json'}")


if __name__ == "__main__":
    main()
