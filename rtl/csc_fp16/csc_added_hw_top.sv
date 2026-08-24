module csc_added_hw_top #(
  parameter int BATCH_WIDTH=8,parameter int INPUT_DEPTH=64,
  parameter int ACC_ENTRIES=64,parameter int COMPARE_W=64,parameter int OUTPUT_DEPTH=64
)(
  input logic clk,input logic rst_n,input logic desc_valid,output logic desc_ready,
  input logic[31:0]desc_column_id,input logic[31:0]desc_x_slot,input logic[31:0]desc_nnz_count,
  input logic[63:0]desc_value_offset,input logic[63:0]desc_index_offset,
  output logic req_valid,input logic req_ready,output logic[1:0]req_type,output logic[63:0]req_addr,
  output logic[7:0]req_tag,input logic rsp_valid,input logic[7:0]rsp_tag,
  output logic pim_mul_valid,input logic pim_mul_ready,output logic[255:0]pim_mul_vector,
  output logic[15:0]pim_mul_scalar,output logic[15:0]pim_mul_lane_mask,
  input logic[255:0]operand_vector,input logic[15:0]operand_scalar,
  input logic[511:0]row_indices,input logic[15:0]operand_lane_mask,
  input logic pim_mul_result_valid,input logic[255:0]pim_mul_result,
  output logic bga_add_req,input logic bga_add_ready,output logic[15:0]bga_add_a,
  output logic[15:0]bga_add_b,input logic bga_add_result_valid,input logic[15:0]bga_add_result,
  input logic drain_req,output logic drain_done,input logic transport_flush,
  input logic producer_done,input logic start_readback,
  output logic write_req_valid,input logic write_req_ready,output logic[63:0]write_req_addr,
  output logic[255:0]write_req_data,output logic[7:0]write_req_tag,
  input logic write_done_valid,input logic[7:0]write_done_tag,
  output logic read_req_valid,input logic read_req_ready,output logic[63:0]read_req_addr,
  output logic[7:0]read_req_tag,input logic read_rsp_valid,input logic[255:0]read_rsp_data,
  input logic[7:0]read_rsp_tag,output logic read_data_valid,input logic read_data_ready,
  output logic[255:0]read_data,output logic writeback_done,output logic readback_done
);
  csc_bg_engine #(.BATCH_WIDTH(BATCH_WIDTH),.INPUT_DEPTH(INPUT_DEPTH),.ACC_ENTRIES(ACC_ENTRIES),
    .COMPARE_W(COMPARE_W),.OUTPUT_DEPTH(OUTPUT_DEPTH)) bg(.*);
endmodule
