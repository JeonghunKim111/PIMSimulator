module csc_bg_engine #(
  parameter int BATCH_WIDTH=8, parameter int INPUT_DEPTH=64,
  parameter int ACC_ENTRIES=64, parameter int COMPARE_W=64,
  parameter int OUTPUT_DEPTH=64
) (
  input logic clk,input logic rst_n,
  input logic desc_valid,output logic desc_ready,input logic[31:0]desc_column_id,
  input logic[31:0]desc_x_slot,input logic[31:0]desc_nnz_count,
  input logic[63:0]desc_value_offset,input logic[63:0]desc_index_offset,
  output logic req_valid,input logic req_ready,output logic[1:0]req_type,
  output logic[63:0]req_addr,output logic[7:0]req_tag,
  input logic rsp_valid,input logic[7:0]rsp_tag,
  output logic pim_mul_valid,input logic pim_mul_ready,
  output logic[255:0]pim_mul_vector,output logic[15:0]pim_mul_scalar,
  output logic[15:0]pim_mul_lane_mask,
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
  logic operands_ready,mul_pending,mul_result_accept,peg_result_ready;
  logic pe_valid,pe_ready;logic[31:0]pe_row;logic[15:0]pe_value;
  logic [BATCH_WIDTH*32-1:0] asm_rows;logic[BATCH_WIDTH*16-1:0]asm_values;
  logic[$clog2(BATCH_WIDTH+1)-1:0]asm_count;logic batch_pending,batch_ready;
  logic bi_valid,bi_ready;logic[31:0]bi_row;logic[15:0]bi_value;
  logic bo_valid,bo_ready;logic[31:0]bo_row;logic[15:0]bo_value;
  logic [$clog2(ACC_ENTRIES+1)-1:0]occ;
  csc_control_top ctrl(.clk,.rst_n,.desc_valid,.desc_ready,.desc_column_id,.desc_x_slot,
    .desc_nnz_count,.desc_value_offset,.desc_index_offset,.req_valid,.req_ready,.req_type,
    .req_addr,.req_tag,.rsp_valid,.rsp_tag,.chunk_consumed(mul_result_accept),.operands_ready);
  assign pim_mul_valid=operands_ready&&!mul_pending;
  assign pim_mul_vector=operand_vector;assign pim_mul_scalar=operand_scalar;
  assign pim_mul_lane_mask=operand_lane_mask;
  assign mul_result_accept=pim_mul_result_valid&&peg_result_ready;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)mul_pending<=0;
    else begin
      if(pim_mul_valid&&pim_mul_ready)mul_pending<=1;
      if(mul_result_accept)mul_pending<=0;
    end
  end
  csc_partial_event_gen peg(.clk,.rst_n,.mul_result_valid(pim_mul_result_valid),
    .mul_result_ready(peg_result_ready),.mul_result(pim_mul_result),.row_indices,.lane_mask(operand_lane_mask),
    .event_valid(pe_valid),.event_ready(pe_ready),.event_row_idx(pe_row),.event_value(pe_value));
  assign pe_ready=!batch_pending;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin asm_count<=0;batch_pending<=0;asm_rows<='0;asm_values<='0;end
    else begin
      if(pe_valid&&pe_ready)begin
        asm_rows[asm_count*32+:32]<=pe_row;asm_values[asm_count*16+:16]<=pe_value;
        if(asm_count==BATCH_WIDTH-1)begin asm_count<=BATCH_WIDTH;batch_pending<=1;end
        else asm_count<=asm_count+1'b1;
      end
      if(!pe_valid&&asm_count!=0&&!batch_pending)batch_pending<=1;
      if(batch_pending&&batch_ready)begin batch_pending<=0;asm_count<=0;end
    end
  end
  csc_batch_ingress #(.BATCH_WIDTH(BATCH_WIDTH)) ingress(.clk,.rst_n,
    .batch_valid(batch_pending),.batch_ready,.batch_count(asm_count),.batch_rows(asm_rows),
    .batch_values(asm_values),.event_valid(bi_valid),.event_ready(bi_ready),
    .event_row(bi_row),.event_value(bi_value));
  csc_bga #(.ACC_ENTRIES(ACC_ENTRIES),.COMPARE_W(COMPARE_W),.INPUT_DEPTH(INPUT_DEPTH),
    .OUTPUT_DEPTH(OUTPUT_DEPTH),.BATCH_WIDTH(BATCH_WIDTH)) bga(.clk,.rst_n,
    .in_valid(bi_valid),.in_ready(bi_ready),.in_row(bi_row),.in_value(bi_value),
    .drain_req,.drain_done,.add_req_valid(bga_add_req),.add_req_ready(bga_add_ready),
    .add_a(bga_add_a),.add_b(bga_add_b),.add_result_valid(bga_add_result_valid),
    .add_result(bga_add_result),.out_valid(bo_valid),.out_ready(bo_ready),
    .out_row(bo_row),.out_value(bo_value),.occupancy(occ));
  csc_transport_top transport(.clk,.rst_n,.in_valid(bo_valid),.in_ready(bo_ready),
    .in_row(bo_row),.in_value(bo_value),.flush(transport_flush),.producer_done,.start_readback,
    .write_req_valid,.write_req_ready,.write_req_addr,.write_req_data,.write_req_tag,
    .write_done_valid,.write_done_tag,.read_req_valid,.read_req_ready,.read_req_addr,
    .read_req_tag,.read_rsp_valid,.read_rsp_data,.read_rsp_tag,.read_data_valid,
    .read_data_ready,.read_data,.writeback_done,.readback_done);
endmodule
