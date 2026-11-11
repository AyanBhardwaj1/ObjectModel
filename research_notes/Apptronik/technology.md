# Apptronik: technology, products, partners, public engineering work (as of Nov 2026)

Research date: 2026-11-11. Method: web search only. Direct page fetches (apptronik.com, therobotreport.com) were blocked by this session's network, so most claims rest on search-result snippets of the cited URLs. One exception: the public MuJoCo model, which I cloned and loaded locally. Labels used below: **[inference]** is my reasoning, not a sourced fact; **[weak]** means only aggregator or blog sources back the claim.

## 1. Company and timeline

- **2016:** Apptronik spun out of UT Austin's Human Centered Robotics Lab. The co-founders worked on NASA's Valkyrie (DARPA Robotics Challenge era) on actuators and controls, and Apptronik says its linear actuators were used throughout Valkyrie ([apptronik.com/our-work](https://apptronik.com/our-work), [TI case study](https://www.ti.com/es-mx/about-ti/company/case-study/apptronik.html)). Leadership: Jeff Cardenas (CEO) and Dr. Nick Paine (CTO). Apollo came out of building 10–15 earlier robots; sources disagree on the count. One earlier robot is Draco, whose page mentions "open access to low-level control software and libraries for RT Linux/ROS integration" ([our-work](https://apptronik.com/our-work)).
- **Aug 2023:** Apollo unveiled ([Robotics 24/7](https://robotics247.com/article/apptronik_humanoid_robot_apptronik_designed_transform_nature_work/news)).
- **Mar 2024:** Mercedes-Benz pilot agreement. **Jun 2024:** multi-phase R&D initiative with GXO, starting with lab trials ([GXO](https://gxo.com/de/news_article/gxo-announces-multi-phase-r-and-d-initiative-with-apptronik), [Robotics 24/7](https://www.robotics247.com/article/gxo_heralds_partnership_with_apptronik_on_an_ai_enabled_humanoid_robot)).
- **Dec 2024:** strategic partnership with the Google DeepMind robotics team ([Robotics 24/7](https://www.robotics247.com/article/apptronik_google_deepmind_robotics_partner_to_accelerate_advancement_of_ai_powered_humanoids), [Maginative](https://www.maginative.com/article/google-deepmind-partners-with-apptronik-to-advance-humanoid-robotics-ambitions/)).
- **Feb 13, 2025:** $350M Series A co-led by B Capital and Capital Factory, with Google participating ([TechCrunch](https://techcrunch.com/2025/02/13/apptronik-raises-350m-to-build-humanoid-robots-with-help-from-google/)). The round was later extended to about $403M, or $415M by other accounts.
- **Feb 2025:** Jabil deal. Jabil will pilot Apollo in its factories and manufacture Apollo for series production ([Robot Report](https://www.therobotreport.com/apptronik-collaborates-with-jabil-to-produce-apollo-humanoid-robots/), [TechCrunch](https://techcrunch.com/?p=2971261)).
- **Mar 2025:** Mercedes-Benz takes an equity stake, reported as a double-digit-million-euro sum. Apollo is tested at the Berlin-Marienfelde Digital Factory Campus and at Kecskemét, Hungary, with fewer than ten units across both sites ([The Star/Reuters](https://www.thestar.com.my/tech/tech-news/2025/03/18/mercedes-benz-takes-stake-in-robotics-maker-apptronik-tests-robots-in-factories), [heise](https://heise.de/-10320691)).
- **Jun 24, 2025:** Elevate Robotics launched as a wholly owned subsidiary for "superhuman," non-humanoid industrial automation, with Paul Hvass as CEO ([Robot Report](https://www.therobotreport.com/apptronik-launches-elevate-robotics-to-build-non-humanoid-robots/)).
- **Sep 25, 2025:** Gemini Robotics 1.5 is trained across ALOHA, bi-arm Franka and Apollo, and DeepMind shows skills transferring between those robots ([DeepMind blog](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/), [arXiv 2510.03342](https://arxiv.org/pdf/2510.03342v3)).
- **Feb 11, 2026:** $520M Series A-X extension, bringing the Series A above $935M and total funding near $1B. Reported valuation is above $5B. Investors include B Capital, Google, Mercedes-Benz, PEAK6, AT&T Ventures, John Deere and QIA ([Robotics 24/7](https://www.robotics247.com/article/apptronik-closes-over-935m-series-a-with-new-520m-extension-round), [Built In](https://builtin.com/articles/apptronik-raises-520m-series-a-x-20260211)).
- **Jun 30, 2026:** Apollo 2 unveiled alongside "Robot Park," an Austin data-collection and training facility of about 90,000 sq ft ([GlobeNewswire](https://www.globenewswire.com/news-release/2026/06/30/3319598/0/en/welcome-to-robot-park-where-apptronik-s-apollo-goes-to-work-training-the-next-generation-of-humanoid-robot-intelligence.html), [Robotics 24/7](https://www.robotics247.com/article/apptronik-unveils-apollo-2-humanoid-robot-opens-robot-park-data-collection-and-training-facility)).
- **Jul 30, 2026:** Google DeepMind's Gemini Robotics 2 adds whole-body control, with Apollo 2 as the featured humanoid ([Robot Report](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/), [Robotics & Automation News](https://roboticsandautomationnews.com/2026/07/31/google-deepmind-unveils-gemini-robotics-2-as-apptronik-humanoid-demonstrates-whole-body-ai/)).

## 2. Apollo hardware

| Item | Value | Source |
|---|---|---|
| Height / weight | 5'8" (about 1.73 m) / about 160 lb (about 73 kg) | Apptronik figures, repeated by [aggregators](https://humanoidapplications.com/robot/apptronik-apollo/) |
| Payload | 55 lb (25 kg) | same |
| Runtime | about 4 h per battery pack; packs are hot-swappable | same |
| Actuators | Proprietary electric linear and rotary actuators. Company claims include fewer parts and higher speed. Force control is "similar to collaborative robots" | [Robotics 24/7](https://robotics247.com/article/apptronik_humanoid_robot_apptronik_designed_transform_nature_work/news), [TI](https://www.ti.com/es-mx/about-ti/company/case-study/apptronik.html) |
| Safety | Force/torque sensing; perimeter zones that slow or stop the robot as people approach; impact detection that halts it. TI supplies motor-control, power and functional-safety parts | same [weak for zone details] |
| Apollo 2 | Comes in bipedal and wheeled-base versions. Apptronik has published no full spec sheet. Third-party DoF and hand figures conflict: 35 DoF with 12-DoF hands, versus a 22-DoF "SharpaWave" hand. Gemini Robotics 2 demos used Inspire hands | [humanoid.guide](https://humanoid.guide/product/apollo-2/), [Venture Atlas](https://www.ventureatlas.org/company/apptronik), [eesel/DeepMind numbers](https://www.eesel.ai/blog/gemini-robotics-2) [weak] |

**Public simulation model (verified locally).** Google DeepMind's MuJoCo Menagerie includes `apptronik_apollo` under the **Apache-2.0** license ([repo folder](https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo)). It was first released 2024-12-20 and last changed 2026-11-02. The changelog notes an IMU correction made "based on Apptronik request" and a change that moved kv gains into joint damping "for better sim2real transfer." The model has a floating base and **32 position actuators**: torso 3, neck 3, 7 per arm and 6 per leg. The fingers are not actuated. Total mass is about 80.9 kg and the timestep is 5 ms. Force limits include ±414 N·m on torso roll and ±114 N·m on shoulder and elbow flexion.

## 3. Target tasks and deployment status

- **Mercedes-Benz:** moving components and kits to the production line, plus quality inspection. Status: pilot ([Reuters via The Star](https://www.thestar.com.my/tech/tech-news/2025/03/18/mercedes-benz-takes-stake-in-robotics-maker-apptronik-tests-robots-in-factories); TechCrunch reported in 2025 that it had "yet to graduate beyond the pilot stage").
- **GXO:** warehouse tasks, starting in a lab. There is no confirmed public figure for production volume. By contrast, Agility has published that Digit moved more than 100k totes at GXO ([Agility](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes)).
- **Jabil:** inspection, sorting, lineside delivery, fixture placement, and moving parts ([Robot Report](https://www.therobotreport.com/apptronik-collaborates-with-jabil-to-produce-apollo-humanoid-robots/)).
- **Trailer unloading, palletizing, case picking, machine tending:** I found these listed only in blog sources [weak].

## 4. Software and AI approach

- **Data engine:** Apollo 2 fleets at Robot Parks and customer sites produce training data through **teleoperation and autonomous execution**. Physics simulation is a third source, used mainly for hardware design and algorithm work. This data trains and refines **Gemini Robotics** models ([GlobeNewswire](https://www.globenewswire.com/news-release/2026/06/30/3319598/0/en/welcome-to-robot-park-where-apptronik-s-apollo-goes-to-work-training-the-next-generation-of-humanoid-robot-intelligence.html), [Interesting Engineering](https://interestingengineering.com/ai-robotics/apptronik-robot-park-humanoid-ai-development)). Apptronik has filed a trademark application for "ROBOT PARK" ([markinton](https://markinton.com/trademark/robot-park-99062086)).
- **Gemini Robotics 2** is a whole-body VLA, an ER 2 reasoning model (available in Google AI Studio) and an on-device model. DeepMind reports these Apollo 2 whole-body pick success rates: table 68.4%, shelf 76.3%, floor 45.7%. DeepMind itself says multi-finger dexterity "remains challenging." The release also showed Apollo 2 working alongside a Franka F3 Duo ([eesel summary](https://www.eesel.ai/blog/gemini-robotics-2), [Robot Report](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)).
- **Tooling, from job postings:** C++ and Python, ROS/ROS 2, Ignition Gazebo, Pinocchio, **MuJoCo**, OMPL, MoveIt, OpenCV, depth cameras, unit and integration testing, and RL sim-to-real work ([Greenhouse posting](https://job-boards.greenhouse.io/apptronik/jobs/5813869004), [Jointaro](https://www.jointaro.com/jobs/apptronik/software-engineering-intern-db08dca9), [Built In](https://builtin.com/company/apptronik): also lists GitLab and Docker). NVIDIA lists Apptronik in its humanoid and simulation ecosystem ([NVIDIA](https://www.nvidia.com/en-us/use-cases/humanoid-robots)). I found no public Apptronik talk about Isaac Sim.
- **Open source:** I found **no official Apptronik GitHub org or repositories**. `gh search repos apptronik` returns only third-party projects. The only quasi-official public asset is the Menagerie model, which DeepMind publishes.

## 5. Hard or open problems relevant to Apptronik's use cases

Sourced:
- Dexterous multi-finger manipulation, and picking objects from the floor, where whole-body success is lowest ([DeepMind numbers via eesel](https://www.eesel.ai/blog/gemini-robotics-2)).
- Moving from pilots to scaled production. Neither Mercedes-Benz nor GXO has published throughput figures ([TechCrunch](https://techcrunch.com/?p=2971261)).
- Perception robustness under varied lighting and object placement, which Agility names for tote work ([Agility](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes)).
- Making a cross-embodiment model work on new robots ([GR 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)).

**[inference]** None of the sources quantifies these, but the target tasks suggest they are hard:
- articulated objects in work cells, such as tote lids, bin flaps, cabinet doors and fixtures;
- generalizing to new SKUs and packaging;
- digital twins of work cells, to evaluate policies before a site goes live;
- turning teleoperated demonstrations into reusable task and scene descriptions;
- sim-to-real calibration (the Menagerie damping and IMU fixes point to this);
- safety monitoring around people.

## 6. Brand and IP cautions

- APPTRONIK is a registered or pending USPTO word mark, and ROBOT PARK is pending ([gleanmark](https://gleanmark.com/trademark/uspto-97391649)). I found **no confirmed APOLLO registration**, but treat "Apollo" as Apptronik's product name anyway. "Gemini" and "Gemini Robotics" are Google marks.
- **[inference]** Guidance for a portfolio project:
  - Do not name it "Apollo-something."
  - Do not use Apptronik logos.
  - Say "compatible with the public MuJoCo Menagerie `apptronik_apollo` model (Apache-2.0)" rather than implying an affiliation.
  - Keep the Apache-2.0 attribution.
  - Do not use renders or press imagery as if they were your own.
