# Controlled data access

## Public and controlled boundary

The public release contains code, configurations, schemas, aggregate reference
results, and checksums. It contains no photograph, direct identifier,
identifier crosswalk, individual clinical record, case-level prediction,
reader-level record, or historical study-model checkpoint.

The following materials may be considered only through controlled access:

- the pseudonymized individual measurement table and repeat tracings;
- the frozen eligible-cohort partition;
- case-level calibration and internal-test predictions;
- the coded operator-experience mapping;
- trained study-model weights.

The minimum controlled bundle documented in `reference/README.md` does not
automatically include trained weights, historical training histories, or
prediction archives for the learning arms. Those materials require a separate
request and approval when they are needed. Possession of the minimum bundle must
not be interpreted as evidence that any optional material was supplied.

When separately approved historical weights are supplied, the public inference
loader supports the validated five-member `c4b` checkpoint layout as well as
checkpoints written by this release. No compatibility claim is made for another
historical arm or checkpoint format. Technical compatibility neither grants
access nor changes the data-use agreement.

Facial photographs are identifiable and are not offered for distribution.
Reader-level data are also withheld. Controlled records are pseudonymized, not
anonymous; the combination of detailed craniofacial measurements may be close
to unique. The CC BY-NC 4.0 license on public aggregate files does not apply to
controlled data.

## Ethics and consent context

The study was reviewed under West China Hospital of Stomatology institutional
review identifier `WCHSIRB-D-2024-362`. Patients had provided broad written
informed consent at the initial visit for research use of records and images,
and the committee waived additional study-specific consent. The cohort included
minors, with a minimum age of seven years. These facts do not create a
public-data authorization and do not override applicable institutional,
ethical, legal, or contractual review.

## Request route

Direct a request to the corresponding authors at their institutional addresses:
Wenli Lai (`wenlilai@scu.edu.cn`) or Hu Long (`hlong@scu.edu.cn`). This release
does not declare a dedicated data-access mailbox. Do not send controlled data,
photographs, or case codes by email.

Approval is not automatic. It may require review by the data-holding
institution, confirmation of the proposed purpose and safeguards, and execution
of a data-use agreement. A requesting institution may also require its own
ethics or legal determination.

## Request template

Do not include patient information, photographs, clinical case codes, or other
sensitive material in the initial request.

```text
Subject: Controlled-access request for the face2ceph study

Applicant and principal investigator:
Institution and department:
Current institutional contact channel:
Project title and concise scientific aim:
Protocol or analysis plan:
Requested data elements or study weights:
Why each requested element is necessary:
Population scope, including whether minors are in scope:
Local ethics determination and approval identifier, if applicable:
Named team members and access roles:
Computing location and jurisdiction:
Security, access-control, and incident-response measures:
Planned linkage to other data, if any:
Retention period and destruction plan:
Planned outputs, publications, and sharing:
Funding and conflicts of interest:
Requested access period:
```

## Expected governance boundary

The exact agreement is determined by the responsible institutions. Applicants
should be prepared for an agreement that, at minimum:

- limits use to the approved scientific purpose, named personnel, location, and
  access period;
- requires least-privilege access, secure storage, and prompt incident
  reporting;
- prohibits re-identification, patient contact, reconstruction of identity or
  facial appearance, and attempts to recover an identifier crosswalk;
- prohibits unapproved linkage, upload to external services, onward transfer,
  publication of individual records, and redistribution of study weights;
- applies data minimization appropriate to a cohort that includes minors;
- requires review of disclosure risk in outputs;
- specifies retention, return, verified destruction, and any audit duties.

This list describes a minimum governance expectation for a future agreement; it
does not assert terms that have not been executed. Authorized users remain
responsible for the agreement, institutional policy, consent limits, and
applicable law. Locally generated normalized images, partitions, predictions,
calibration state, and checkpoints remain controlled derivatives and must not be
published merely because the code that created them is public.
