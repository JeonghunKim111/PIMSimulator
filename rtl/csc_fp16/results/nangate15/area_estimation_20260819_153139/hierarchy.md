# Elaborated instance hierarchy

```text
csc_added_hw_top [1]
└── csc_bg_engine [1, parameterized Q64]
    ├── csc_control_top [1]
    │   ├── csc_descriptor_engine [1]
    │   ├── csc_agu [1, TAG_W=8]
    │   └── csc_request_tracker [1, DEPTH=16, TAG_W=8]
    ├── csc_partial_event_gen [1]
    ├── csc_batch_ingress [1, BATCH_WIDTH=8]
    ├── csc_bga [1, ACC/COMPARE/INPUT/OUTPUT=64]
    └── csc_transport_top [1]
        ├── csc_result_serializer [1]
        ├── csc_burst_packer [1]
        └── csc_result_transport_ctrl [1]
```

Every instance count is one within the current top. The top itself represents one BG support engine; multi-BG replication is not present in this RTL.
