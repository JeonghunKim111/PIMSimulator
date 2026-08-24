module csc_control_top #(
  parameter int TRACK_DEPTH=16, parameter int TAG_W=8
) (
  input logic clk,input logic rst_n,input logic desc_valid,output logic desc_ready,
  input logic [31:0] desc_column_id,input logic [31:0] desc_x_slot,input logic [31:0] desc_nnz_count,
  input logic [63:0] desc_value_offset,input logic [63:0] desc_index_offset,
  output logic req_valid,input logic req_ready,output logic [1:0] req_type,
  output logic [63:0] req_addr,output logic [TAG_W-1:0] req_tag,
  input logic rsp_valid,input logic [TAG_W-1:0] rsp_tag,input logic chunk_consumed,
  output logic operands_ready
);
  logic chunk_valid,desc_chunk_ready,agu_chunk_ready,busy,done,chunk_launched;
  logic[31:0]column,xslot,chunk;
  logic[63:0]vc,ic;logic[15:0]mask;logic[4:0]lanes;logic tr_ready,match;logic[1:0]rtype;
  logic ix,iv,il,ih,issue_done,agu_req_valid;
  csc_descriptor_engine d(.clk,.rst_n,.desc_valid,.desc_ready,.desc_column_id,.desc_x_slot,
    .desc_nnz_count,.desc_value_offset,.desc_index_offset,.chunk_valid,.chunk_ready(desc_chunk_ready),
    .column_id(column),.x_slot(xslot),.value_cursor(vc),.index_cursor(ic),
    .lane_mask(mask),.valid_lanes(lanes),.chunk_id(chunk),.descriptor_done(done),.busy(busy));
  assign desc_chunk_ready=chunk_launched&&chunk_consumed;
  always_ff @(posedge clk or negedge rst_n)begin
    if(!rst_n)chunk_launched<=0;
    else begin
      if(chunk_valid&&!chunk_launched&&agu_chunk_ready)chunk_launched<=1;
      if(desc_chunk_ready)chunk_launched<=0;
    end
  end
  csc_agu #(.TAG_W(TAG_W)) a(.clk,.rst_n,.chunk_valid(chunk_valid&&!chunk_launched),
    .chunk_ready(agu_chunk_ready),.column_id(column),.x_slot(xslot),
    .value_cursor(vc),.index_cursor(ic),.valid_lanes(lanes),.chunk_id(chunk),.req_valid(agu_req_valid),
    .req_ready(req_ready&&tr_ready),.req_type,.req_addr,.req_tag,.issued_x(ix),.issued_value(iv),
    .issued_index_low(il),.issued_index_high(ih),.issue_group_done(issue_done));
  assign req_valid=agu_req_valid&&tr_ready;
  csc_request_tracker #(.DEPTH(TRACK_DEPTH),.TAG_W(TAG_W)) t(.clk,.rst_n,
    .alloc_valid(agu_req_valid&&req_ready&&tr_ready),.alloc_ready(tr_ready),.alloc_tag(req_tag),.alloc_type(req_type),
    .alloc_last(issue_done),
    .rsp_valid,.rsp_tag,.rsp_match(match),.rsp_type(rtype),.operands_ready,.outstanding_count());
endmodule
