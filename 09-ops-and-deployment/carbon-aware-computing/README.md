---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Carbon-aware computing

Training and running AI uses a lot of electricity, and how much carbon that electricity emits depends on where and when it's used. Carbon-aware computing moves flexible work to cleaner places and times.

**Roles:** ML engineer, platform engineer; leaders interested in sustainability.
**Prerequisites:** basic cloud knowledge

## Key concepts

- **Carbon intensity** is the grams of CO₂ emitted per kWh of electricity. It varies a lot by region and by hour, with the mix of coal, gas, wind, solar and hydro ([L2](L2_Explore_Carbon_Intensity.ipynb)).
- **Move work in space.** Training in a low-carbon cloud region can cut emissions several times over for the same job ([L3](L3_Training_Models.ipynb)).
- **Move work in time.** Batch jobs can wait for hours when the grid is cleaner, using real-time and forecast data ([L4](L4_Using_Real_Time.ipynb)).
- **Measure your footprint.** Cloud providers and tools such as CodeCarbon estimate the emissions of your workloads ([L5](L5_GC_Footprint.ipynb)).
- **Efficiency helps too.** Smaller models, quantization and better batching reduce energy per request as well as cost.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Carbon aware computing for GenAI developers](https://www.deeplearning.ai/short-courses/carbon-aware-computing-for-genai-developers/) (Google Cloud) | Short course | Beginner | Free when reviewed |
| [Electricity Maps](https://www.electricitymaps.com/) | Data | All levels | Free tier |
| [CodeCarbon](https://codecarbon.io/) | Tool | Intermediate | Free |
| [Green Software Foundation](https://greensoftware.foundation/) | Community | All levels | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2](L2_Explore_Carbon_Intensity.ipynb) | Exploring live carbon intensity with the Electricity Maps API |
| [L3](L3_Training_Models.ipynb) | Training models in low-carbon regions on Google Cloud |
| [L4](L4_Using_Real_Time.ipynb) | Choosing a region with real-time energy data for a Vertex AI training job |
| [L5](L5_GC_Footprint.ipynb) | Understanding your Google Cloud carbon footprint |

## Next steps

- [AI governance](../../08-evaluation-and-safety/ai-governance/)
- [Serving LLMs efficiently](../efficiently-serving-llms/)
