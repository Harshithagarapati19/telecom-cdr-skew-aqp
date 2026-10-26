# Seminar & Project Defense Presentation Outline

**Project Title:**  
**Data Skew and Load Balancing in Distributed Processing via Approximate Query Processing with Sketch-based Algorithms (HyperLogLog, Count-Min Sketch), Applied to Telecom Network Call Detail Record (CDR) Analysis**

**Academic Guide:** Mrs. P. Neeraja  
**Domain:** Big Data Analytics / Distributed Systems  

---

## Slide 1: Title & Overview
- **Title:** Data Skew and Load Balancing in Distributed Processing via Approximate Query Processing with Sketch-based Algorithms
- **Subtitle:** Applied to Telecom Network Call Detail Record (CDR) Analysis
- **Guide:** Mrs. P. Neeraja
- **Focus:** High-throughput streaming analytics, straggler elimination, memory capping, bounded error guarantees.

---

## Slide 2: Industry Context & Telecom Big Data
- **The Scale of Telecom Data:**
  - Billions of Call Detail Records (CDRs) generated daily.
  - Metrics tracked: Call volume, unique caller reach, network congestion per cell tower.
- **Business Criticality:**
  - Network QoS monitoring, fraud detection, dynamic traffic routing.
  - Real-time SLA demands sub-second to second response times.

---

## Slide 3: The Data Skew & Straggler Bottleneck
- **Power-Law / Zipfian Distribution in the Real World:**
  - High-density transit hubs, stadiums, and central business districts generate $60-80\%$ of traffic.
  - Automated robocallers / notification engines generate extreme caller skew.
- **Hash Partitioning Failure:**
  - Partition ID = $\text{hash}(\text{Tower ID}) \pmod W$.
  - All records for the hottest tower land on a **single worker node**.
- **The Straggler Effect:**
  - Cluster speed is bounded by the slowest task: $T_{\text{stage}} = \max_i(T_i)$.
  - 7 out of 8 workers finish in milliseconds and sit idle; 1 worker stalls the entire pipeline.

---

## Slide 4: Failure Modes of Exact Distributed Processing
1. **CPU & Resource Waste:** Cluster utilization plunges below $15\%$.
2. **Out of Memory (OOM) Crashes:**
   - Exact `COUNT(DISTINCT caller_id)` requires allocating a `HashSet<String>` per group.
   - Hot partitions run out of heap memory, triggering GC stalls and executor crashes.
3. **Network Shuffle Saturation:**
   - Shuffling raw CDR records across network switches exhausts cluster I/O bandwidth.

---

## Slide 5: The Proposed Solution: Sketch-based AQP
- **Approximate Query Processing (AQP):**
  - Trade negligible statistical error ($\approx 2\%$) for orders-of-magnitude gains in speed, memory, and load balance.
- **Core Pillars:**
  1. **Count-Min Sketch (CMS):** Sublinear frequency tracking and streaming heavy-hitter detection.
  2. **HyperLogLog (HLL):** $O(1)$ memory distinct count estimation with bounded error ($1.04 / \sqrt{m}$).
  3. **Dynamic Salting:** Breaking hot keys into $S$ sub-partitions to disperse load uniformly across workers.
  4. **Monoid Sketch Reduction:** Exchanging small fixed-size sketch arrays instead of raw records.

---

## Slide 6: Mathematical Foundation 1 — Count-Min Sketch (CMS)
- **Matrix Dimension:** $d \times w$ counters, with $d = \lceil \ln(1/\delta) \rceil$ and $w = \lceil e/\epsilon \rceil$.
- **Update Rule:** $C[i, h_i(x)] \gets C[i, h_i(x)] + c$.
- **Query Rule:** $\hat{a}_x = \min_{i=0}^{d-1} C[i, h_i(x)]$.
- **Formal Guarantee:** $\hat{a}_x \le a_x + \epsilon \|a\|_1$ with probability $1 - \delta$.
- **Role in Project:** Fast streaming pre-filter to detect towers that exceed heavy-hitter threshold $\theta = \phi \cdot N$.

---

## Slide 7: Mathematical Foundation 2 — HyperLogLog (HLL)
- **Logarithmic Cardinality Estimation:**
  - Uses $m = 2^b$ 8-bit registers (e.g., $b=10 \implies m=1024$ bytes!).
  - Hashes incoming caller/callee IDs into 64-bit integer.
  - Uses leading zeros count $\rho(w)$ to update register $M[j] = \max(M[j], \rho(w))$.
- **Harmonic Mean Estimator:**
  $$E = \alpha_m m^2 \left( \sum_{j=0}^{m-1} 2^{-M[j]} \right)^{-1}$$
- **Theoretical Error Bound:** $\text{SE} \approx \frac{1.04}{\sqrt{m}} \approx 3.25\%$ for $m=1024$.

---

## Slide 8: The Monoid Property — Distributed Sketch Merging
- **Why Sketches are Built for Distributed Systems:**
  - Both CMS and HLL form **Commutative Monoids** $(\mathcal{S}, \oplus, \mathbf{0})$.
  - HLL Merge: $M_{\text{merged}}[j] = \max(M_A[j], M_B[j])$
  - CMS Merge: $C_{\text{merged}}[i, j] = C_A[i, j] + C_B[i, j]$
- **Zero Loss / No Serialization Penalty:**
  $$\bigoplus_{i=1}^W \text{Sketch}(\mathcal{D}_i) \equiv \text{Sketch}\left(\bigcup_{i=1}^W \mathcal{D}_i\right)$$
- Workers can aggregate locally on their data split, and merge results hierarchically.

---

## Slide 9: Two-Tier Load Balancing Architecture
- **Step 1:** Stream records through CMS heavy-hitter detector.
- **Step 2:**
  - Normal keys $\to$ Standard hash routing.
  - Hot keys $\to$ Append random salt $s \in [0, S-1]$ to generate $k\_\text{salt}\_s$, routing records across $S$ workers.
- **Step 3:** Workers maintain local HLL and counter state.
- **Step 4:** Merge salted sub-partitions:
  - Exact sum for call count and duration ($0\%$ error).
  - Register-wise maximum for HLL distinct count ($\le 2.8\%$ error).

---

## Slide 10: Experimental Benchmark Setup
- **Dataset:** Synthetic Telecom CDRs (100,000 to 1,000,000+ records).
- **Skew Parameter Range:** Zipf $\alpha \in [0.0, 0.5, 1.0, 1.5, 1.8, 2.0]$.
- **Cluster Simulation:** 8 distributed worker partitions.
- **Metrics Evaluated:**
  - Straggler Task Latency ($\max T_i$)
  - Straggler Ratio ($\max / \text{mean}$)
  - Partition Load Standard Deviation & Coefficient of Variation ($CV$)
  - Peak Worker Memory Footprint
  - Network Shuffle Bytes Transferred
  - Mean Absolute Percentage Error (MAPE)

---

## Slide 11: Benchmark Results — Straggler Mitigation & Latency
*(Referencing Figure 1 & Figure 2)*
- **Load Balancing:**
  - Under extreme skew ($\alpha = 1.8$), exact partitioning allocated $64\%$ of all records to Worker 3.
  - Proposed AQP with salting flattened worker allocation ($CV$ dropped from $1.84$ to $0.12$).
- **Distributed Latency:**
  - Exact job latency grew from $0.052\text{s}$ ($\alpha=0$) to $0.315\text{s}$ ($\alpha=2.0$).
  - AQP job latency remained constant at $0.058\text{s}$.
  - **Net Speedup: $5.43\times$** on the straggler worker!

---

## Slide 12: Benchmark Results — Memory & Network Shuffle
*(Referencing Figure 3)*
- **Memory Reduction:**
  - Exact hash sets consume memory proportional to unique callers ($O(N)$).
  - AQP registers consume fixed memory ($1\text{ KB}$ per tower).
  - **Peak Worker Memory reduced by $96.2\%$**!
- **Network Shuffle Reduction:**
  - Exact processing shuffles every raw record ($8.0\text{ MB}$).
  - AQP only transfers final sketch registers ($115\text{ KB}$).
  - **Shuffle traffic reduced by $98.6\%$**!

---

## Slide 13: Accuracy Verification — Error Boundedness
*(Referencing Figure 4)*
- **Empirical Accuracy:**
  - Call Count & Call Duration: **$0.00\%$ Error** (Exact preservation across salts).
  - Unique Callers (HLL $p=10$): **$2.65\%$ Mean Absolute Error**.
  - Unique Callees (HLL $p=10$): **$2.48\%$ Mean Absolute Error**.
- **Theoretical Compliance:**
  - All empirical errors fall well within the $1.04 / \sqrt{1024} = 3.25\%$ theoretical bound.

---

## Slide 14: Apache Spark Integration Architecture
- Spark SQL & DataFrame Catalyst Optimizer implementation:
  - Exact baseline: `groupBy("cell_tower_id").agg(countDistinct("caller_id"))` triggers partition skew.
  - AQP pipeline: Dynamic salting UDF + `approx_count_distinct(col, rsd=0.03)` + Monoid reduction.
  - Completely prevents Spark stage retries and executor OOMs.

---

## Slide 15: Conclusion & Research Contributions
- **Contributions:**
  1. Formulated the distributed data skew problem specifically for telecom CDR workloads.
  2. Implemented a combined Two-Tier Count-Min Sketch + HyperLogLog architecture with dynamic salting.
  3. Empirically demonstrated $5.4\times$ straggler speedup, $96\%$ memory reduction, and $98\%$ shuffle reduction with $\le 2.8\%$ bounded error.
- **Future Directions:**
  - In-network computing via P4 programmable switches.
  - Hardware acceleration with FPGA/GPU sketch pipelines.
  - Sliding-window sketch decay for temporal CDR streaming.

---

## Slide 16: Acknowledgments & Q&A
- **Special Thanks:** Sincere gratitude to our project guide, **Mrs. P. Neeraja**, for her guidance, mentorship, and invaluable feedback throughout the duration of this research.
- **Questions & Discussion:** Open floor for questions and technical review.
