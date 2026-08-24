package csc_fp16_pkg;
  localparam string CSC_FP16_BATCH8_Q64_DEFAULT = "CSC_FP16_BATCH8_Q64_DEFAULT";
  localparam int BATCH_WIDTH_DEFAULT  = 8;
  localparam int INPUT_DEPTH_DEFAULT  = 64;
  localparam int ACC_ENTRIES_DEFAULT  = 64;
  localparam int COMPARE_W_DEFAULT    = 64;
  localparam int OUTPUT_DEPTH_DEFAULT = 64;
  localparam int ROW_W_DEFAULT        = 32;
  localparam int VALUE_W_DEFAULT      = 16;
  localparam int GLOBAL_BG_COUNT      = 64;
  typedef enum logic [1:0] {REQ_X, REQ_VALUE, REQ_INDEX_LOW, REQ_INDEX_HIGH} req_type_t;
  typedef struct packed {logic [31:0] row_idx; logic [15:0] value;} partial_t;
  typedef struct packed {
    logic [31:0] column_id;
    logic [31:0] x_slot;
    logic [31:0] nnz_count;
    logic [63:0] value_offset;
    logic [63:0] index_offset;
    logic [31:0] flags;
  } descriptor_t;
endpackage
