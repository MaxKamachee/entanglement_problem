# Defensive reference buckets — selection record

Keep set: **ACADEMIC, NIST, RFC, VENDOR**. PATENT and MITRE_CAR are dropped (legalese / stub). `ATTACK_XREF` (attack.mitre.org) is dropped per the no-crosswalks constraint; mis-bucketed patents (patentimages/patentguru) are reclassified into PATENT. **OTHER: curated keep** — substantive refs kept, noise hosts dropped via `configs/corpus.yaml:defense_other_drop_hosts`. **Curated defensive selection: 141 distinct URLs.**

## Per-bucket counts

| bucket | refs | status |
|---|---|---|
| PATENT | 148 | DROP |
| OTHER | 95 | CURATED-KEEP |
| MITRE_CAR | 89 | DROP |
| VENDOR | 24 | KEEP |
| ACADEMIC | 15 | KEEP |
| NIST | 11 | KEEP |
| RFC | 10 | KEEP |
| ATTACK_XREF | 2 | DROP |

## OTHER sample (50 of 95)

| url | title | supports concept |
|---|---|---|
| https://docs.librenms.org/Extensions/Network-Map | Libre NMS - Network Map Extension | Network Mapping |
| https://ssh.com/academy/iam/password-key-rotation | Password and Key Rotation | Certificate Rotation |
| https://web.archive.org/web/20070510153306/http://www.fwtk.org/fwtk/docs/documentation.html | FWTK Documentation | Inbound Traffic Filtering |
| https://csis.gmu.edu/noel/pubs/2016_NATO_IST_148.pdf | Mission Dependency Modeling for Cyber Situational Awareness | Operational Dependency Mapping |
| https://stigviewer.com/stig/windows_10 | Windows 10 Security Technical Implementation Guide | Application Configuration Hardening |
| https://nebelwelt.net/files/15LangSec.pdf | The Correctness-Security Gap in Compiler Optimization | Dead Code Elimination |
| https://trustedcomputinggroup.org/wp-content/uploads/TCG_TNC_TAP_Use_Cases_v1r0p35_published.pdf | Trusted Attestation Protocol Use Cases |  |
| https://prosoft-technology.com/prosoft/download/9671/182665/file/PLX3x_UserManual | PLX3x Series Multi-Protocol Gateways | OT Variable Access Restriction |
| https://nrc.gov/docs/ml0932/ml093290424.pdf | Technical Product Guide Tricon Systems | Disable Remote Access |
| https://docs.tia.siemens.cloud/r/simatic_s7_1200_manual_collection_enus_20/programming-concepts/using-blocks-to-structure-your-program/data-block-db | S7-1200 Programmable controller | OT Variable Access Restriction |
| https://cisa.gov/resources-tools/resources/cisa-cpg-checklist | CISA CPG Checklist | Change Default Password |
| http://biometric-solutions.com/keystroke-dynamics.html | Keystroke Dynamics | Input Device Analysis |
| https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html | Secrets Management Cheat Sheet | Credential Scrubbing |
| https://cwe.mitre.org/data/definitions/476.html | CWE-476: NULL Pointer Dereference | Null Pointer Checking |
| https://en.wikipedia.org/wiki/Passive_infrared_sensor | Passive infrared sensor | Motion Sensor Monitoring |
| https://github.com/wesleyraptor/streamingphish | StreamingPhish | Passive Certificate Analysis |
| https://gnu.org/software/c-intro-and-ref/manual/html_node/Pointer-Arithmetic.html | Pointer Arithmetic in C | Memory Block Start Validation |
| https://stigviewer.com/stig/red_hat_enterprise_linux_8 | Red Hat Enterprise Linux 8 Security Technical Implementation Guide | Application Configuration Hardening |
| https://cs.princeton.edu/courses/archive/fall98/cs441/mainus/node4.html | Why type checking? | Variable Type Validation |
| https://qualcomm.com/media/documents/files/whitepaper-pointer-authentication-on-armv8-3.pdf | Pointer Authentication on ARMv8.3 | Pointer Authentication |
| https://media.defense.gov/2021/Aug/03/2002820425/-1/-1/0/CTR_Kubernetes_Hardening_Guidance_1.1_20220315.PDF | Kubernetes Hardening Guide | Container Image Analysis |
| https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1048.1241 | Transparent ROP Exploit Mitigation using Indirect Branch Tracing | Indirect Branch Call Analysis |
| https://mirrors.edge.kernel.org/pub/linux/utils/cryptsetup/LUKS_docs/on-disk-format.pdf | LUKS1 On-Disk Format SpecificationVersion 1.2.3 | Disk Encryption |
| http://detect-respond.blogspot.com/2013/03/the-pyramid-of-pain.html | The Pyramid of Pain | Identifier Activity Analysis |
| https://ic3.gov/CSA/2022/220325.pdf | TRITON Malware Remains Threat to Global Critical Infrastructure Industrial Contr | Operating Mode Monitoring |
| https://cwe.mitre.org/documents/sources/TheCLASPApplicationSecurityProcess.pdf | The CLASP Application Security Process | Reference Nullification |
| https://github.com/Neo23x0/munin | Online Hash Checker for Virustotal and Other Services |  |
| https://web.archive.org/web/20180407204216/https://isc.sans.edu/diary/Decoy+Personas+for+Safeguarding+Online+Identity+Using+Deception/16159 | Decoy Personas for Safeguarding Online Identity Using Deception | Decoy Persona |
| http://nsl.cs.columbia.edu/projects/minestrone/papers/Symbiotes.pdf | Defending Embedded Systems with Software Symbiotes | Firmware Embedded Monitoring Code |
| https://owasp.org/www-community/controls/Certificate_and_Public_Key_Pinning | Certificate and Public Key Pinning | Certificate Pinning |
| https://mitre.org/research/technology-transfer/technology-licensing/cyber-command-system-cycs | Cyber Command System (CYCS) | Operational Dependency Mapping |
| https://eprints.qut.edu.au/21172/1/21172.pdf | Network-Based Buffer Overflow Detection by Exploit Code Analysis | Byte Sequence Emulation |
| https://uefi.org/sites/default/files/resources/PI_Spec_1_7_A_final_May1.pdf | UEFI Platform Initialization (PI) Specification | Bootloader Authentication |
| https://sans.org/cyber-security-courses/managing-enterprise-cloud-security-vulnerabilities | MGT516: Managing Security Vulnerabilities: Enterprise and Cloud | Operational Risk Assessment |
| https://networkworld.com/article/3331199/what-does-aslr-do-for-linux.html | How ASLR protects Linux systems from buffer overflow attacks | Segment Address Offset Randomization |
| https://yokogawa.com/us/library/resources/faqs/pressure-what-is-hardware-write-protect | What is Hardware Write Protect? | Hardware-based Write Protection |
| https://docs.librenms.org/Extensions/Oxidized | LibreNMSDocs - Oxidized Extension | Disk Encryption |
| https://omg.org/spec/UAF | Unified Architecture Framework (UAF) | Data Exchange Mapping |
| https://dni.gov/files/Governance/IC-Tech-Specs-for-Const-and-Mgmt-of-SCIFs-v15.pdf | Technical Specifications for Construction and Management of Sensitive Compartmen | RF Shielding |
| https://w3.org/TR/webauthn-2 | Web Authentication: An API for accessing Public Key Credentials
Level 2 | Credential Transmission Scoping |
| https://citeseerx.ist.psu.edu/document?doi=bf4d34a6f9d0168bb07433e84c1567bbe1ba8188 | Understanding the Domain Registration Behavior of Spammers | Domain Registration Takedown |
| https://cs.umd.edu/~jkatz/security/downloads/passwords_revealed-weir.pdf | Testing Metrics for Password Creation Policies by Attacking Large Sets of Reveal | Strong Password Policy |
| https://cwe.mitre.org/data/definitions/457.html | CWE-457: Use of Uninitialized Variable | Variable Initialization |
| https://github.com/osquery/osquery/blob/d2be385d71f401c85872f00d479df8f499164c5a/osquery/tables/system/windows/users.cpp | OS Query Windows User Collection Code |  |
| https://blogs.gartner.com/john_pescatore/2008/10/02/this-week-in-network-security-history-the-firewall-toolkit | FWTK - Firewall Toolkit |  |
| https://en.wikipedia.org/wiki/Network_mapping | Network Mapping |  |
| https://hacks.mozilla.org/2021/05/introducing-firefox-new-site-isolation-security-architecture | Site Isolation Design Document | Application-based Process Isolation |
| https://docs.device42.com/auto-discovery/network-auto-discovery | SNMP - Network Auto Discovery | Active Logical Link Mapping |
| https://sap-press.com/organizational-management-in-sap-erp-hcm_3996 | Organization Mapping in SAP ERP HCM | Organization Mapping |
| https://trustedcomputinggroup.org/resource/tpm-library-specification | TPM 2.0 Library Specification | TPM Boot Integrity |
