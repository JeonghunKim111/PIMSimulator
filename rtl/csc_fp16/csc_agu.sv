module csc_agu #(
  parameter int ADDR_W = 64, parameter int TAG_W = 8,
  parameter logic [ADDR_W-1:0] VALUE_BASE = 64'h0,
  parameter logic [ADDR_W-1:0] INDEX_BASE = 64'h0000_0000_0002_0000,
  parameter logic [ADDR_W-1:0] X_BASE = 64'h0000_0000_0004_0000
) (
  input logic clk, input logic rst_n,
  input logic chunk_valid, output logic chunk_ready,
  input logic [31:0] column_id, input logic [31:0] x_slot,
  input logic [63:0] value_cursor, input logic [63:0] index_cursor,
  input logic [4:0] valid_lanes, input logic [31:0] chunk_id,
  output logic req_valid, input logic req_ready,
  output logic [1:0] req_type, output logic [ADDR_W-1:0] req_addr,
  output logic [TAG_W-1:0] req_tag,
  output logic issued_x, output logic issued_value,
  output logic issued_index_low, output logic issued_index_high,
  output logic issue_group_done
);
  typedef enum logic [2:0] {IDLE, ISSUE_X, ISSUE_VALUE, ISSUE_IL, ISSUE_IH} state_t;
  state_t state;
  logic need_high;
  assign req_valid = state != IDLE;
  assign req_tag = {chunk_id[TAG_W-3:0], req_type};
  assign chunk_ready = (state == IDLE);
  assign issue_group_done=(state==ISSUE_IH)||((state==ISSUE_IL)&&!need_high);
  always_comb begin
    req_type = 2'd0; req_addr = '0;
    case (state)
      ISSUE_X: begin req_type=2'd0; req_addr=X_BASE + ({32'b0,column_id} << 1); end
      ISSUE_VALUE: begin req_type=2'd1; req_addr=VALUE_BASE + value_cursor; end
      ISSUE_IL: begin req_type=2'd2; req_addr=INDEX_BASE + index_cursor; end
      ISSUE_IH: begin req_type=2'd3; req_addr=INDEX_BASE + index_cursor + 32; end
      default: begin end
    endcase
    req_addr[4:0] = 5'b0;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state<=IDLE; need_high<=1'b0; issued_x<=0; issued_value<=0;
      issued_index_low<=0; issued_index_high<=0;
    end else begin
      issued_x<=0; issued_value<=0; issued_index_low<=0; issued_index_high<=0;
      if (state==IDLE && chunk_valid) begin state<=ISSUE_X; need_high <= valid_lanes > 8; end
      else if (req_valid && req_ready) begin
        case(state)
          ISSUE_X: begin issued_x<=1; state<=ISSUE_VALUE; end
          ISSUE_VALUE: begin issued_value<=1; state<=ISSUE_IL; end
          ISSUE_IL: begin issued_index_low<=1;state<=need_high?ISSUE_IH:IDLE;end
          ISSUE_IH: begin issued_index_high<=1;state<=IDLE;end
          default: state<=IDLE;
        endcase
      end
    end
  end
endmodule
