"""Generate the DOCX files in sample_docs/. Run once; output is committed."""
from pathlib import Path
import docx

HERE = Path(__file__).parent


def build(name, intro, sections):
    d = docx.Document()
    d.add_heading(name.replace("_", " ").replace(".docx", "").title(), level=0)
    if intro:
        d.add_paragraph(intro)
    for h, paragraphs in sections:
        d.add_heading(h, level=1)
        for p in paragraphs:
            d.add_paragraph(p)
    d.save(str(HERE / name))
    print("wrote", name)


build(
    "hr_policies.docx",
    "This handbook describes the people policies that apply to all Nimbus Robotics "
    "employees. It is maintained by the People Operations team and reviewed annually. "
    "Where a local law provides a greater benefit than this policy, the local law applies.",
    [
        ("Paid Time Off", [
            "Full-time employees accrue 15 days of paid time off per year, accrued monthly and "
            "available for use as it accrues. PTO must be requested at least two weeks in advance "
            "and approved by the direct manager, except in cases of illness or emergency.",
            "Unused PTO may be carried over up to a maximum of five days into the following year; "
            "any balance above that is forfeited unless local law requires otherwise. Employees are "
            "encouraged to take at least one full week of consecutive time off each year to rest.",
        ]),
        ("Sick Leave", [
            "Employees receive ten paid sick days per year, separate from paid time off. Sick leave "
            "may be used for the employee's own illness or to care for an immediate family member.",
            "A doctor's note is only required for absences longer than three consecutive days. "
            "Unused sick leave does not carry over and is not paid out on departure.",
        ]),
        ("Remote Work Policy", [
            "Employees are expected to work from the office at least three days per week, with the "
            "specific days coordinated within each team to ensure adequate in-person collaboration.",
            "Fully remote arrangements require VP approval and are reviewed annually. Remote workers "
            "must maintain a reliable internet connection and a distraction-free environment for "
            "meetings. The company provides a one-time home-office stipend for remote-eligible staff.",
        ]),
        ("Expense Reimbursement", [
            "Business expenses under 100 dollars may be approved by a direct manager. Expenses at or "
            "above that threshold require finance approval before they are incurred where practical.",
            "All expense reports must include itemized receipts and be submitted within 30 days of "
            "the expense. Reimbursements are paid in the next payroll cycle after approval.",
        ]),
        ("Performance Reviews", [
            "Formal performance reviews are conducted annually in December, with a lighter-weight "
            "check-in at mid-year. Reviews assess both results and how those results were achieved.",
            "Compensation adjustments resulting from a review take effect at the start of the next "
            "fiscal year. Promotions may occur at either cycle when the case is clearly met.",
        ]),
        ("Parental Leave", [
            "New parents receive 12 weeks of fully paid parental leave, available to all parents "
            "regardless of gender and usable any time within the first year after birth or adoption.",
            "Leave may be taken continuously or, with manager agreement, in two separate blocks. "
            "Returning parents are entitled to a gradual ramp-up schedule for their first two weeks.",
        ]),
        ("Benefits Enrollment", [
            "Employees must enroll in benefits within 30 days of their start date; those who miss the "
            "window wait until the next open enrollment in November. The package includes health, "
            "dental, and vision coverage, with the company covering the majority of the premium.",
            "A retirement savings plan with company matching is available after ninety days of "
            "employment. Life and disability insurance are provided automatically at no cost.",
        ]),
    ],
)

build(
    "product_roadmap.docx",
    "This roadmap summarizes the planned direction of the Nimbus product line. Dates are "
    "targets, not commitments, and are revisited at the start of each quarter. Customer-"
    "facing commitments are made only through signed agreements, not this document.",
    [
        ("Supported Platforms", [
            "The Nimbus control app currently supports Windows and macOS. Mobile support is not yet "
            "available but is under evaluation for a future release.",
            "The control app requires a 64-bit operating system and a network connection to the "
            "robot controller. A browser-based dashboard is available for read-only monitoring.",
        ]),
        ("Launch Schedule", [
            "The Nimbus 2.0 firmware is scheduled to launch on September 15th, following a public "
            "beta that opens one month before launch. The beta is limited to existing customers who "
            "opt in and agree to provide structured feedback.",
            "Subsequent point releases follow a six-week cadence. Each release is gated on a quality "
            "review and a successful soak period on internal hardware.",
        ]),
        ("Hardware Compatibility", [
            "Nimbus 2.0 is compatible with the Atlas and Orion robot arms. Legacy Pioneer arms are "
            "supported until end of year, after which they move to security-only maintenance.",
            "A compatibility checker in the control app verifies firmware and accessory versions "
            "before an upgrade. Unsupported accessory combinations are blocked with a clear message.",
        ]),
        ("Upcoming Features", [
            "Planned work includes adaptive grip-force control, a visual teach-by-demonstration mode, "
            "and an expanded library of pre-built motion templates for common assembly tasks.",
            "A fleet-management view for customers operating multiple arms is in early design. "
            "Features are sequenced by customer demand, safety impact, and engineering readiness.",
        ]),
        ("Deprecation Policy", [
            "Features slated for removal are announced at least two releases in advance with a "
            "migration path. Deprecated APIs continue to function during the announced window.",
            "Customers on deprecated functionality receive direct outreach from their account team "
            "well before the removal date to ensure a smooth transition.",
        ]),
    ],
)

build(
    "it_policy.docx",
    "This policy governs the acceptable use and management of company information technology. "
    "It applies to all employees and contractors who use Nimbus devices, accounts, or networks. "
    "The IT team administers this policy and assists with any questions about compliance.",
    [
        ("Acceptable Use", [
            "Company devices and accounts are provided for business purposes; incidental personal use "
            "is permitted as long as it does not interfere with work or violate any policy.",
            "Employees must not use company systems to store illegal content, run unauthorized "
            "businesses, or bypass security controls. All activity may be monitored consistent with law.",
        ]),
        ("Device Encryption", [
            "All company laptops must have full-disk encryption enabled before being issued, and the "
            "recovery key is escrowed with IT. Devices automatically lock after five minutes idle.",
            "External drives that store company data must also be encrypted. Unencrypted removable "
            "media is not permitted for confidential or restricted information.",
        ]),
        ("Software Approval", [
            "New software must be approved by IT before installation on company devices to ensure it "
            "is licensed, supported, and free of known security issues. A catalog of pre-approved "
            "applications is available for self-service installation.",
            "Browser extensions and command-line tools are treated as software and follow the same "
            "approval process. Unapproved software found on a device may be removed automatically.",
        ]),
        ("Account Provisioning and Deprovisioning", [
            "New employee accounts are provisioned within two business days of the start date, with "
            "access scoped to the role. Managers request any additional access through the IT portal.",
            "When an employee leaves, all access is revoked on their last day, and devices are "
            "returned and wiped. Shared resources owned by the departing employee are reassigned.",
        ]),
        ("Network and Wi-Fi", [
            "The corporate network is segmented so that guest, employee, and production traffic are "
            "isolated. Guests use a separate guest network that cannot reach internal systems.",
            "Connecting personal servers, routers, or wireless access points to the corporate network "
            "is prohibited. Remote access to internal systems requires the company VPN and MFA.",
        ]),
        ("Data Backup and Recovery", [
            "Business-critical systems are backed up daily, with backups tested for restorability on a "
            "monthly basis. Employees are responsible for storing work files in approved, backed-up "
            "locations rather than only on local disks.",
            "Recovery time and recovery point objectives are defined per system and reviewed annually. "
            "Personal devices are not backed up by the company.",
        ]),
    ],
)

print("done")
