module csc_request_tracker #(
  parameter int DEPTH=16, parameter int TAG_W=8
) (
  input logic clk, input logic rst_n,
  input logic alloc_valid, output logic alloc_ready,
  input logic [TAG_W-1:0] alloc_tag, input logic [1:0] alloc_type,input logic alloc_last,
  input logic rsp_valid, input logic [TAG_W-1:0] rsp_tag,
  output logic rsp_match, output logic [1:0] rsp_type,
  output logic operands_ready, output logic [$clog2(DEPTH+1)-1:0] outstanding_count
);
  logic [DEPTH-1:0] valid;
  logic [TAG_W-1:0] tags [DEPTH];
  logic [1:0] types [DEPTH];
  logic [3:0] ready_mask;logic sealed;
  integer i; integer free_idx; integer match_idx;
  always_comb begin
    free_idx=-1; match_idx=-1;
    for(i=0;i<DEPTH;i=i+1) begin
      if(!valid[i] && free_idx<0) free_idx=i;
      if(valid[i] && tags[i]==rsp_tag && match_idx<0) match_idx=i;
    end
    alloc_ready=(free_idx>=0); rsp_match=rsp_valid && (match_idx>=0);
    rsp_type=(match_idx>=0)?types[match_idx]:2'b0;
    operands_ready=sealed&&(outstanding_count==0);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin valid<='0; ready_mask<='0; outstanding_count<='0;sealed<=0; end
    else begin
      if(alloc_valid && alloc_ready) begin
        valid[free_idx]<=1; tags[free_idx]<=alloc_tag; types[free_idx]<=alloc_type;
        if(alloc_last)sealed<=1;
      end
      if(rsp_match) begin
        valid[match_idx]<=0; ready_mask[rsp_type]<=1;
      end
      case ({alloc_valid&&alloc_ready,rsp_match})
        2'b10: outstanding_count<=outstanding_count+1'b1;
        2'b01: outstanding_count<=outstanding_count-1'b1;
        default: begin end
      endcase
      if(operands_ready)begin ready_mask<='0;sealed<=0;end
    end
  end
endmodule
